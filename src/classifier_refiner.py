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
        Refines a single detection's class label based on geometric priors, 
        aspect-ratio constraints, and perspective depth scale.
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
        # Motorcycles and bicycles are vertically elongated or have a compact footprint.
        # Upright rider profile: aspect_ratio is typically <= 0.75, rarely > 0.95.
        # ----------------------------------------------------------------------
        if raw_class_name == "car":
            # If predicted as 'car' but is vertically tall (narrow width) or very small
            is_narrow_vertical = aspect_ratio <= 0.68
            is_compact_footprint = (area < expected_car_area * 0.42 and aspect_ratio <= 0.85)
            is_extreme_vertical = aspect_ratio <= 0.55

            if is_narrow_vertical or is_compact_footprint or is_extreme_vertical:
                refined_class = "motorcycle"
                refined_conf = max(confidence, 0.72)

        elif raw_class_name in ("motorcycle", "bicycle"):
            # If YOLO predicted motorcycle but box is broad horizontally and bulky, it is a car
            if aspect_ratio >= 1.10 and area > expected_car_area * 0.65:
                refined_class = "car"
                refined_conf = max(confidence, 0.70)
            else:
                refined_class = "motorcycle"

        # ----------------------------------------------------------------------
        # RULE 2: BUS DISAMBIGUATION
        # Buses are passenger transit vehicles with long rectangular bodies (high aspect ratio)
        # or massive vertical box profile.
        # ----------------------------------------------------------------------
        if raw_class_name == "bus":
            # If predicted as bus but too small to be one, reclassify to car or van
            if area < expected_car_area * 1.35 and aspect_ratio < 1.70:
                refined_class = "car"
                refined_conf = confidence * 0.90
        elif raw_class_name == "car":
            # If predicted as car but has huge footprint with long profile typical of a transit bus
            if area > expected_car_area * 2.2 and aspect_ratio >= 1.65:
                refined_class = "bus"
                refined_conf = max(confidence, 0.75)

        # ----------------------------------------------------------------------
        # RULE 3: TRUCK VS CAR / BUS DISAMBIGUATION
        # Trucks have high volume, box cargo beds, trailers, or high cabins.
        # Compact SUVs / hatchbacks / sedans should NOT be labeled trucks.
        # ----------------------------------------------------------------------
        if raw_class_name == "truck":
            # Standard passenger car sizes falsely flagged as truck
            is_passenger_scale = area < expected_car_area * 1.70
            is_compact_shape = (0.65 <= aspect_ratio <= 1.50)

            if is_passenger_scale and is_compact_shape:
                refined_class = "car"
                refined_conf = confidence * 0.90
            elif area > expected_car_area * 2.5 and aspect_ratio >= 1.80:
                # Extra-long multi-window or transit shape is often bus
                if h < (img_h * 0.35 * depth_factor):
                    refined_class = "bus"

        elif raw_class_name == "car":
            # Massive freight truck falsely labeled as car
            is_freight_scale = area > expected_car_area * 2.6
            is_very_tall_cabin = h > (img_h * 0.25 * depth_factor) and area > expected_car_area * 1.8
            is_long_freight_trailer = aspect_ratio > 1.90 and area > expected_car_area * 1.9

            if is_freight_scale or is_very_tall_cabin or is_long_freight_trailer:
                refined_class = "truck"
                refined_conf = max(confidence, 0.70)

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
