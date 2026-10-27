"""
RoadSentry AI - Vehicle Classifier Refinement Module
Stabilizes and corrects vehicle classifications (car, motorcycle, bus, truck)
using geometric aspect-ratio analysis, relative scale profiling, and temporal track voting.
"""

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np


class VehicleClassifierRefiner:
    """
    Refines YOLOv8 vehicle predictions to resolve common CCTV misclassifications:
      1. Bikes/motorcycles misclassified as cars (narrow aspect ratio, small footprint).
      2. Cars misclassified as trucks (passenger-sized vehicles falsely flagged as heavy freight).
      3. Trucks misclassified as cars (large cargo/box trucks falsely flagged as passenger cars).
      4. Single-frame flickering between car <-> truck <-> motorcycle.
    """

    COCO_TO_STANDARD = {
        1: "motorcycle",  # COCO bicycle -> motorcycle/two-wheeler
        2: "car",         # COCO car
        3: "motorcycle",  # COCO motorcycle
        5: "bus",         # COCO bus
        7: "truck"        # COCO truck
    }

    VALID_CLASSES = ["car", "motorcycle", "bus", "truck"]

    def __init__(
        self,
        temporal_window: int = 15,
        decay_factor: float = 0.95,
        min_votes_to_switch: int = 3
    ):
        self.temporal_window = temporal_window
        self.decay_factor = decay_factor
        self.min_votes_to_switch = min_votes_to_switch

        # Persistent track votes: track_id -> {class_name: weighted_score}
        self.track_class_scores = defaultdict(lambda: defaultdict(float))
        # Stable confirmed class: track_id -> class_name
        self.stable_classes = {}

    def refine_single_box(
        self,
        bbox: List[int],
        raw_class_name: str,
        confidence: float,
        frame_shape: Tuple[int, int, int],
        all_bboxes: Optional[List[List[int]]] = None
    ) -> Tuple[str, float]:
        """
        Refines a single detection's class label based on geometric priors and relative scene context.
        """
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h
        aspect_ratio = w / float(h)  # width / height

        img_h, img_w = frame_shape[:2]
        cy = (y1 + y2) / 2.0
        # Normalized vertical depth in frame (0.0 at top/horizon, 1.0 at bottom/near camera)
        depth_factor = np.clip(cy / float(img_h), 0.1, 1.0)

        # Calibrated expected passenger car area at this perspective depth (CCTV geometry)
        # Vehicles near top (horizon) are smaller, vehicles near bottom (foreground) are larger
        expected_car_area = (img_w * img_h * 0.018) * (0.30 + 0.70 * (depth_factor ** 1.4))

        refined_class = raw_class_name
        refined_conf = confidence

        # ----------------------------------------------------------------------
        # RULE 1: BIKE / MOTORCYCLE DISAMBIGUATION
        # Motorcycles and bicycles are vertically elongated (narrow width, upright rider).
        # A true car in CCTV traffic almost never has aspect_ratio <= 0.65 unless clipped.
        # ----------------------------------------------------------------------
        if raw_class_name == "car":
            # Narrow vertical profile or significantly smaller than expected car size
            if (aspect_ratio <= 0.62 and area < expected_car_area * 0.75) or (aspect_ratio <= 0.52):
                refined_class = "motorcycle"
                refined_conf = max(confidence, 0.65)
            elif area < expected_car_area * 0.38 and aspect_ratio <= 0.80:
                refined_class = "motorcycle"
                refined_conf = max(confidence, 0.60)

        elif raw_class_name in ("motorcycle", "bicycle"):
            # If YOLO predicted motorcycle but box is wide and large, it is actually a car
            if aspect_ratio >= 1.15 and area > expected_car_area * 0.75:
                refined_class = "car"
                refined_conf = max(confidence, 0.60)
            else:
                refined_class = "motorcycle"

        # ----------------------------------------------------------------------
        # RULE 2: CAR VS TRUCK / BUS DISAMBIGUATION
        # Trucks and buses are physically much larger (large cargo boxes, trailers).
        # Standard passenger sedans, hatchbacks, and compact SUVs falsely labeled as "truck".
        # ----------------------------------------------------------------------
        if raw_class_name == "truck":
            # If the box area is within normal passenger car bounds and aspect ratio is compact
            is_passenger_scale = area < expected_car_area * 1.85
            is_compact_shape = (0.70 <= aspect_ratio <= 1.55)

            if is_passenger_scale and is_compact_shape:
                refined_class = "car"
                refined_conf = confidence * 0.92

        elif raw_class_name == "car":
            # If YOLO predicted "car" on a massive truck, semi-trailer, or large box truck
            is_freight_scale = area > expected_car_area * 2.8
            is_very_tall = h > (img_h * 0.28 * depth_factor) and area > expected_car_area * 2.0
            is_long_trailer = aspect_ratio > 1.95 and area > expected_car_area * 2.0

            if is_freight_scale or is_very_tall or is_long_trailer:
                refined_class = "truck"
                refined_conf = max(confidence, 0.65)

        # ----------------------------------------------------------------------
        # RULE 3: BUS DISAMBIGUATION
        # Buses have high aspect ratio (long side profile) or high volume with large windows.
        # ----------------------------------------------------------------------
        if raw_class_name == "bus":
            if area < expected_car_area * 1.30 and not (aspect_ratio > 1.85):
                # Too small to be a public transit bus; likely a van or passenger car
                refined_class = "car"

        return refined_class, refined_conf

    def update_track_class(
        self,
        track_id: int,
        bbox: List[int],
        raw_class_name: str,
        confidence: float,
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[str, float]:
        """
        Updates the track's temporal voting history and returns the stabilized class label.
        Eliminates single-frame class flickering.
        """
        # Step 1: Geometric prior refinement for current frame
        frame_class, frame_conf = self.refine_single_box(
            bbox=bbox,
            raw_class_name=raw_class_name,
            confidence=confidence,
            frame_shape=frame_shape
        )

        # Step 2: Temporal score accumulation with decay
        scores = self.track_class_scores[track_id]

        # Apply decay to past votes
        for c in list(scores.keys()):
            scores[c] *= self.decay_factor

        # Add current frame evidence
        # Weight by detection confidence and class prior
        scores[frame_class] += float(frame_conf)

        # Step 3: Determine dominant class
        if scores:
            best_class = max(scores.items(), key=lambda kv: kv[1])[0]
        else:
            best_class = frame_class

        # Update stable class
        self.stable_classes[track_id] = best_class

        # Return smoothed confidence
        total_score = sum(scores.values()) if scores else 1.0
        smoothed_conf = (scores[best_class] / total_score) if total_score > 0 else frame_conf

        return best_class, min(0.99, max(0.40, float(smoothed_conf)))

    def clear_inactive_tracks(self, active_track_ids: List[int]):
        """Remove old tracks no longer present in the scene."""
        active_set = set(active_track_ids)
        for tid in list(self.track_class_scores.keys()):
            if tid not in active_set:
                del self.track_class_scores[tid]
                if tid in self.stable_classes:
                    del self.stable_classes[tid]
