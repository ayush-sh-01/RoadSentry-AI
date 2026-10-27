"""
RoadSentry AI - Traffic Density Module
Computes vehicle count and congestion level (Low/Medium/High) globally and per Region of Interest (ROI).
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2


class DensityEstimator:
    """
    Evaluates traffic concentration per frame and inside defined polygonal ROIs.
    """

    LEVEL_COLORS = {
        "LOW": (0, 200, 0),       # Green
        "MEDIUM": (0, 165, 255),  # Orange/Amber
        "HIGH": (0, 0, 255)       # Red
    }

    def __init__(
        self,
        low_thresh: int = 5,
        medium_thresh: int = 12,
        roi_polygon: Optional[List[Tuple[float, float]]] = None
    ):
        self.low_thresh = low_thresh
        self.medium_thresh = medium_thresh
        # Normalized coordinates [ (x, y), ... ] between 0.0 and 1.0
        self.roi_normalized = roi_polygon

    def get_density_level(
        self,
        count: int,
        occupancy_ratio: float = 0.0,
        avg_speed: float = 5.0,
        stoppage_ratio: float = 0.0
    ) -> str:
        """
        Classifies traffic into LOW, MEDIUM, or HIGH based on vehicle count,
        road area occupancy ratio, and crawling/stoppage queue conditions.
        """
        # Condition for HIGH congestion (Traffic Jam / Severe Bottleneck)
        if (
            count > self.medium_thresh or
            occupancy_ratio >= 0.25 or
            (count >= 3 and (occupancy_ratio >= 0.15 or stoppage_ratio >= 0.50)) or
            (count >= 2 and occupancy_ratio >= 0.22 and stoppage_ratio >= 0.40)
        ):
            return "HIGH"

        # Condition for MEDIUM congestion (Moderate Traffic / Queue Forming)
        if (
            count > self.low_thresh or
            occupancy_ratio >= 0.12 or
            (count >= 2 and (stoppage_ratio >= 0.50 or avg_speed < 2.0)) or
            (count >= 1 and occupancy_ratio >= 0.18)
        ):
            return "MEDIUM"

        return "LOW"

    def estimate(
        self,
        frame_shape: Tuple[int, int, int],
        tracked_objects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculates global and ROI vehicle density, area occupancy, and flow speed.
        """
        h, w = frame_shape[:2]
        total_count = len(tracked_objects)

        # Scale ROI polygon to frame dimensions if provided
        roi_pts = None
        roi_count = 0
        vehicles_in_roi = []
        roi_objs = []

        if self.roi_normalized:
            roi_pts = np.array(
                [[int(x * w), int(y * h)] for x, y in self.roi_normalized],
                dtype=np.int32
            )
            roi_area = max(1.0, float(abs(cv2.contourArea(roi_pts))))

            for obj in tracked_objects:
                cx, cy = obj["centroid"]
                if cv2.pointPolygonTest(roi_pts, (float(cx), float(cy)), False) >= 0:
                    roi_count += 1
                    vehicles_in_roi.append(obj["track_id"])
                    roi_objs.append(obj)
        else:
            roi_count = total_count
            vehicles_in_roi = [obj["track_id"] for obj in tracked_objects]
            roi_objs = tracked_objects
            roi_area = float(w * h)

        # Calculate total vehicle bounding box footprint
        occupied_pixels = 0
        speeds = []
        for obj in roi_objs:
            if "bbox" in obj:
                bx1, by1, bx2, by2 = obj["bbox"]
                bw = max(1, bx2 - bx1)
                bh = max(1, by2 - by1)
                occupied_pixels += (bw * bh)
            speeds.append(obj.get("speed", 5.0))

        occupancy_ratio = min(1.0, occupied_pixels / roi_area) if roi_area > 0 else 0.0
        avg_speed = float(np.mean(speeds)) if speeds else 5.0
        stoppage_count = sum(1 for s in speeds if s < 1.5)
        stoppage_ratio = float(stoppage_count / len(speeds)) if speeds else 0.0

        density_level = self.get_density_level(
            count=roi_count,
            occupancy_ratio=occupancy_ratio,
            avg_speed=avg_speed,
            stoppage_ratio=stoppage_ratio
        )

        return {
            "total_count": total_count,
            "roi_count": roi_count,
            "density_level": density_level,
            "vehicles_in_roi": vehicles_in_roi,
            "roi_points": roi_pts,
            "occupancy_ratio": occupancy_ratio,
            "avg_speed": avg_speed,
            "stoppage_ratio": stoppage_ratio
        }

    def draw_roi_and_density(
        self,
        frame: np.ndarray,
        density_data: Dict[str, Any]
    ) -> np.ndarray:
        """Draws ROI boundary and HUD density overlay badge on the frame."""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        roi_pts = density_data.get("roi_points")
        density_level = density_data["density_level"]
        roi_count = density_data["roi_count"]
        total_count = density_data["total_count"]
        occupancy = density_data.get("occupancy_ratio", 0.0) * 100
        avg_speed = density_data.get("avg_speed", 0.0)
        level_color = self.LEVEL_COLORS.get(density_level, (0, 255, 0))

        # Render translucent ROI overlay
        if roi_pts is not None and len(roi_pts) >= 3:
            overlay = annotated.copy()
            cv2.polylines(overlay, [roi_pts], isClosed=True, color=level_color, thickness=2)
            cv2.fillPoly(overlay, [roi_pts], color=level_color)
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

        # Draw HUD Panel at top-right
        hud_w, hud_h = 260, 92
        hud_x = w - hud_w - 15
        hud_y = 15

        # Background card
        cv2.rectangle(annotated, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (20, 20, 20), -1)
        cv2.rectangle(annotated, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), level_color, 2)

        # Text labels
        cv2.putText(annotated, "TRAFFIC DENSITY", (hud_x + 10, hud_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Level badge
        cv2.putText(annotated, f"STATE: {density_level}", (hud_x + 10, hud_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, level_color, 2, cv2.LINE_AA)
        
        # Vehicle count and occupancy stats
        cv2.putText(annotated, f"Vehicles: {roi_count} (Total: {total_count})", (hud_x + 10, hud_y + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        
        flow_status = "Jammed/Stopped" if avg_speed < 1.5 else f"Flow: {avg_speed:.1f}px/f"
        cv2.putText(annotated, f"Occupancy: {occupancy:.0f}% | {flow_status}", (hud_x + 10, hud_y + 83),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 255), 1, cv2.LINE_AA)
        return annotated
