"""
RoadSentry AI - Cumulative Traffic Heatmap Module
Accumulates vehicle centroids over time with exponential temporal decay to highlight persistent bottlenecks.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2


class HeatmapAccumulator:
    """
    Maintains a decaying 2D spatial heatmap matrix of vehicle activity.
    """

    def __init__(
        self,
        decay_factor: float = 0.96,
        gaussian_kernel: int = 31,
        sigma: float = 15.0,
        intensity_boost: float = 2.0
    ):
        self.decay_factor = decay_factor
        self.gaussian_kernel = gaussian_kernel if gaussian_kernel % 2 == 1 else gaussian_kernel + 1
        self.sigma = sigma
        self.intensity_boost = intensity_boost
        self.accumulator: np.ndarray = None

    def initialize(self, shape: Tuple[int, int]):
        """Initialize zeroed float accumulator matrix."""
        h, w = shape[:2]
        self.accumulator = np.zeros((h, w), dtype=np.float32)

    def update(self, frame_shape: Tuple[int, int], tracked_objects: List[Dict[str, Any]]) -> np.ndarray:
        """
        Applies exponential temporal decay and adds vehicle centroids.
        Slow/stopped vehicles deposit higher heat to highlight true congestion bottlenecks.
        """
        h, w = frame_shape[:2]
        if self.accumulator is None or self.accumulator.shape != (h, w):
            self.initialize((h, w))

        # Apply temporal decay (H = H * decay)
        self.accumulator *= self.decay_factor

        # Splat centroids onto a temporary frame buffer
        if tracked_objects:
            point_mask = np.zeros((h, w), dtype=np.float32)
            for obj in tracked_objects:
                cx, cy = obj["centroid"]
                if 0 <= cx < w and 0 <= cy < h:
                    # Vehicles that are stopped or moving very slowly generate much higher heat
                    speed = obj.get("speed", 5.0)
                    stationary = obj.get("stationary_frames", 0)
                    if stationary >= 3 or speed < 1.2:
                        weight = 3.0  # Intense heat for stopped / jammed vehicles
                        radius = 10
                    elif speed < 2.5:
                        weight = 1.8  # Medium heat for crawling traffic
                        radius = 8
                    else:
                        weight = 0.8  # Faint heat for free-flowing passing vehicles
                        radius = 6

                    cv2.circle(point_mask, (cx, cy), radius=radius, color=weight, thickness=-1)

            # Apply Gaussian blur to create smooth heatmap blooms
            blurred_points = cv2.GaussianBlur(
                point_mask,
                (self.gaussian_kernel, self.gaussian_kernel),
                self.sigma
            )
            self.accumulator += blurred_points * self.intensity_boost

        return self.accumulator

    def get_colored_heatmap(self) -> np.ndarray:
        """Returns BGR colorized heatmap using COLORMAP_JET."""
        if self.accumulator is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Normalize to 0-255 range
        max_val = np.max(self.accumulator)
        if max_val > 1e-4:
            norm_map = np.clip((self.accumulator / max_val) * 255.0, 0, 255).astype(np.uint8)
        else:
            norm_map = np.zeros_like(self.accumulator, dtype=np.uint8)

        color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        return color_map

    def blend_overlay(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Alpha blends the colorized heatmap directly onto the original frame."""
        color_map = self.get_colored_heatmap()
        if color_map.shape[:2] != frame.shape[:2]:
            color_map = cv2.resize(color_map, (frame.shape[1], frame.shape[0]))

        # Mask out very low activity zones so they stay transparent
        max_val = np.max(self.accumulator) if self.accumulator is not None else 0
        if max_val > 1e-4:
            activity_mask = (self.accumulator / max_val) > 0.08
            blended = frame.copy()
            overlay_zone = cv2.addWeighted(frame, 1.0 - alpha, color_map, alpha, 0)
            blended[activity_mask] = overlay_zone[activity_mask]
            return blended
        return frame

    def get_annotated_heatmap_view(self, frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """
        Returns a beginner-friendly heatmap view with the live road background
        and an on-screen color spectrum legend explaining congestion severity.
        """
        blended = self.blend_overlay(frame, alpha=alpha)
        h, w = blended.shape[:2]

        # Draw an intuitive on-screen color legend card at the bottom
        bar_w = min(420, int(w * 0.70))
        bar_h = 16
        bar_x = 15
        bar_y = h - 55

        # Background card for readability
        card_w = bar_w + 24
        card_h = 46
        cv2.rectangle(blended, (bar_x - 5, bar_y - 22), (bar_x + card_w, bar_y + bar_h + 10), (18, 20, 28), -1)
        cv2.rectangle(blended, (bar_x - 5, bar_y - 22), (bar_x + card_w, bar_y + bar_h + 10), (50, 55, 75), 1)

        # Legend title
        cv2.putText(
            blended,
            "CONGESTION INTENSITY SCALE:",
            (bar_x, bar_y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (210, 210, 210),
            1,
            cv2.LINE_AA
        )

        # Gradient color bar (from Cyan/Green to Yellow to Red)
        gradient_row = np.linspace(50, 255, bar_w, dtype=np.uint8)
        gradient_2d = np.tile(gradient_row, (bar_h, 1))
        gradient_bgr = cv2.applyColorMap(gradient_2d, cv2.COLORMAP_JET)

        blended[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = gradient_bgr
        cv2.rectangle(blended, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

        # Scale labels
        cv2.putText(blended, "Moving (Low)", (bar_x, bar_y + bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(blended, "Moderate", (bar_x + bar_w // 2 - 25, bar_y + bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(blended, "Heavy Jam / Stopped", (bar_x + bar_w - 110, bar_y + bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 80, 255), 1, cv2.LINE_AA)

        # Hotspot status tag at top-left
        stats = self.get_hotspot_summary()
        tag_color = (0, 200, 0) if stats["bottleneck_level"] == "CLEAR" else (
            (0, 165, 255) if stats["bottleneck_level"] == "MODERATE" else (0, 0, 255)
        )
        status_text = f"HOTSPOT STATUS: {stats['bottleneck_level']} ({stats['bottleneck_desc']})"
        cv2.rectangle(blended, (15, 12), (15 + len(status_text) * 8 + 15, 34), (18, 20, 28), -1)
        cv2.rectangle(blended, (15, 12), (15 + len(status_text) * 8 + 15, 34), tag_color, 1)
        cv2.putText(blended, status_text, (22, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tag_color, 1, cv2.LINE_AA)

        return blended

    def get_hotspot_summary(self) -> Dict[str, Any]:
        """Provides a simple high-level summary of active bottleneck heat."""
        if self.accumulator is None:
            return {"peak_intensity": 0, "bottleneck_level": "CLEAR", "bottleneck_desc": "Road is clear"}

        max_val = float(np.max(self.accumulator))
        # Intensity score 0 to 100
        intensity = min(100, int((max_val / 8.0) * 100)) if max_val > 0 else 0

        if intensity < 25:
            level = "CLEAR"
            desc = "Vehicles flowing smoothly"
        elif intensity < 65:
            level = "MODERATE"
            desc = "Slowdown or queue forming"
        else:
            level = "SEVERE"
            desc = "Persistent stoppage / bottleneck"

        return {
            "peak_intensity": intensity,
            "bottleneck_level": level,
            "bottleneck_desc": desc
        }
