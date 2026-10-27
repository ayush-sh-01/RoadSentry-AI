"""
Unit Tests for Vehicle Classification Refinement and Disambiguation
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.classifier_refiner import VehicleClassifierRefiner


class TestVehicleClassifierRefiner(unittest.TestCase):

    def setUp(self):
        self.refiner = VehicleClassifierRefiner()
        self.frame_shape = (720, 1280, 3)

    def test_narrow_bike_misclassified_as_car(self):
        """A narrow vertical box (e.g., width 40, height 90, aspect ratio ~0.44) should be refined to motorcycle."""
        bbox = [400, 300, 440, 390]  # w=40, h=90, aspect_ratio=0.44
        refined_class, conf = self.refiner.refine_single_box(
            bbox=bbox,
            raw_class_name="car",
            confidence=0.55,
            frame_shape=self.frame_shape
        )
        self.assertEqual(refined_class, "motorcycle")

    def test_passenger_car_misclassified_as_truck(self):
        """A standard passenger car footprint (e.g. w=120, h=90, aspect ratio 1.33) should not be flagged as truck."""
        bbox = [400, 300, 520, 390]  # w=120, h=90, compact area
        refined_class, conf = self.refiner.refine_single_box(
            bbox=bbox,
            raw_class_name="truck",
            confidence=0.60,
            frame_shape=self.frame_shape
        )
        self.assertEqual(refined_class, "car")

    def test_large_freight_vehicle_misclassified_as_car(self):
        """A very large commercial box truck (e.g. w=380, h=280) should be refined to truck."""
        bbox = [300, 200, 680, 480]  # w=380, h=280, massive area
        refined_class, conf = self.refiner.refine_single_box(
            bbox=bbox,
            raw_class_name="car",
            confidence=0.52,
            frame_shape=self.frame_shape
        )
        self.assertEqual(refined_class, "truck")

    def test_bicycle_mapped_to_motorcycle(self):
        """Bicycles should be cleanly standardized under two-wheeler / motorcycle."""
        bbox = [200, 250, 245, 340]
        refined_class, conf = self.refiner.refine_single_box(
            bbox=bbox,
            raw_class_name="bicycle",
            confidence=0.70,
            frame_shape=self.frame_shape
        )
        self.assertEqual(refined_class, "motorcycle")

    def test_temporal_smoothing_prevents_flicker(self):
        """A persistent car should not flip to truck on 1 or 2 outlier noisy frames."""
        track_id = 42
        car_box = [300, 400, 460, 520]

        # Feed 5 frames of car
        for _ in range(5):
            cls, _ = self.refiner.update_track_class(
                track_id=track_id,
                bbox=car_box,
                raw_class_name="car",
                confidence=0.85,
                frame_shape=self.frame_shape
            )
            self.assertEqual(cls, "car")

        # Now feed 1 frame of noisy 'truck' prediction
        noisy_cls, _ = self.refiner.update_track_class(
            track_id=track_id,
            bbox=car_box,
            raw_class_name="truck",
            confidence=0.60,
            frame_shape=self.frame_shape
        )
        # Should stay 'car' thanks to temporal accumulator
        self.assertEqual(noisy_cls, "car")


if __name__ == "__main__":
    unittest.main()
