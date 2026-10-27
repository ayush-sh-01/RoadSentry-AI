"""
Unit and Integration Tests for RoadSentry AI
"""

import os
import sys
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.density import DensityEstimator
from src.heatmap import HeatmapAccumulator
from src.accident_detector import AccidentDetector
from src.emergency_classifier import EmergencyClassifier
from src.logger import IncidentLogger


class TestRoadSentryComponents(unittest.TestCase):

    def test_density_estimator(self):
        estimator = DensityEstimator(low_thresh=3, medium_thresh=8)
        # Test level categorization
        self.assertEqual(estimator.get_density_level(2), "LOW")
        self.assertEqual(estimator.get_density_level(6), "MEDIUM")
        self.assertEqual(estimator.get_density_level(15), "HIGH")

        # Test frame estimation
        fake_tracks = [
            {"track_id": 1, "centroid": (100, 100)},
            {"track_id": 2, "centroid": (200, 200)},
        ]
        res = estimator.estimate((480, 640, 3), fake_tracks)
        self.assertEqual(res["total_count"], 2)
        self.assertEqual(res["density_level"], "LOW")

    def test_heatmap_decay(self):
        heatmap = HeatmapAccumulator(decay_factor=0.5)
        heatmap.initialize((200, 200))
        fake_tracks = [{"centroid": (100, 100)}]
        heatmap.update((200, 200), fake_tracks)

        val1 = np.max(heatmap.accumulator)
        self.assertGreater(val1, 0.0)

        # Update with empty tracks -> should decay by half
        heatmap.update((200, 200), [])
        val2 = np.max(heatmap.accumulator)
        self.assertAlmostEqual(val2, val1 * 0.5, places=3)

    def test_accident_iou_overlap(self):
        detector = AccidentDetector(iou_thresh=0.25, persistence_frames=1)
        # Identical overlapping boxes should have IoU = 1.0
        boxA = [100, 100, 200, 200]
        boxB = [100, 100, 200, 200]
        iou = detector.compute_iou(boxA, boxB)
        self.assertAlmostEqual(iou, 1.0, places=2)

        # Disjoint boxes should have IoU = 0.0
        boxC = [300, 300, 400, 400]
        self.assertAlmostEqual(detector.compute_iou(boxA, boxC), 0.0, places=2)

    def test_single_vehicle_offroad_barrier_crash(self):
        """Verifies single-vehicle off-road boundary breach with sudden halt triggers accident alert."""
        roi = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        detector = AccidentDetector(
            roi_polygon=roi,
            offroad_min_speed=3.0,
            offroad_stop_speed=1.0,
            offroad_persistence_frames=2,
            offroad_cooldown_frames=10
        )
        frame_shape = (100, 100, 3)

        # 1. Vehicle starts inside road with high speed (centroid at 50, 50 -> normalized (0.5, 0.5) is inside)
        track_inside = {
            "track_id": 7,
            "bbox": [40, 40, 60, 60],
            "class_name": "car",
            "centroid": (50, 50),
            "speed": 5.5,
            "stationary_frames": 0
        }
        detector.analyze_frame([track_inside], frame_idx=1, frame_shape=frame_shape)
        self.assertTrue(detector.track_was_inside[7])

        # 2. Vehicle breaches boundary and crashes into side railing (centroid at 90, 50 -> outside ROI)
        track_crashed = {
            "track_id": 7,
            "bbox": [80, 40, 100, 60],
            "class_name": "car",
            "centroid": (90, 50),
            "speed": 0.2,
            "stationary_frames": 3
        }

        # Frame 2: Anomaly persists
        alerts1 = detector.analyze_frame([track_crashed], frame_idx=2, frame_shape=frame_shape)
        self.assertEqual(len(alerts1), 0)  # Needs persistence_frames=2

        # Frame 3: Sustained anomaly triggers OFFROAD_BARRIER_CRASH
        alerts2 = detector.analyze_frame([track_crashed], frame_idx=3, frame_shape=frame_shape)
        self.assertEqual(len(alerts2), 1)
        self.assertEqual(alerts2[0]["type"], "OFFROAD_BARRIER_CRASH")
        self.assertEqual(alerts2[0]["vehicle_ids"], [7])

    def test_normal_driving_does_not_trigger_offroad_crash(self):
        """Verifies normal driving (high speed inside road or moving off-road) does not trigger false alerts."""
        roi = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        detector = AccidentDetector(roi_polygon=roi, offroad_persistence_frames=2)
        frame_shape = (100, 100, 3)

        # Fast moving car inside road
        track_fast = {
            "track_id": 8,
            "bbox": [40, 40, 60, 60],
            "class_name": "car",
            "centroid": (50, 50),
            "speed": 6.0,
            "stationary_frames": 0
        }
        for f in range(5):
            alerts = detector.analyze_frame([track_fast], frame_idx=f, frame_shape=frame_shape)
            self.assertEqual(len(alerts), 0)

    def test_missing_roi_safely_handled(self):
        """Verifies detector handles unconfigured/None road ROI safely without errors."""
        detector = AccidentDetector(roi_polygon=None)
        track = {
            "track_id": 9,
            "bbox": [40, 40, 60, 60],
            "class_name": "car",
            "centroid": (50, 50),
            "speed": 5.0,
            "stationary_frames": 0
        }
        alerts = detector.analyze_frame([track], frame_idx=1, frame_shape=(100, 100, 3))
        self.assertEqual(len(alerts), 0)

    def test_emergency_beacon_heuristic(self):
        classifier = EmergencyClassifier()
        # Synthetic ambulance roof with bright red and blue pixels
        crop = np.zeros((60, 60, 3), dtype=np.uint8)
        # Roof area (first 20 rows) with red and blue
        crop[5:15, 5:25] = [0, 0, 255]    # Bright red in BGR
        crop[5:15, 30:50] = [255, 0, 0]   # Bright blue in BGR

        is_emer, conf, kind = classifier.check_roof_beacon_heuristic(crop)
        self.assertTrue(is_emer)
        self.assertGreater(conf, 0.5)

    def test_civilian_blue_car_rejected(self):
        classifier = EmergencyClassifier()
        # Solid blue civilian car (both roof and body are blue)
        crop = np.zeros((80, 80, 3), dtype=np.uint8)
        crop[:, :] = [220, 50, 50]  # Blue in BGR

        is_emer, conf, kind = classifier.check_roof_beacon_heuristic(crop)
        self.assertFalse(is_emer)
        self.assertEqual(kind, "civilian")

    def test_civilian_red_car_rejected(self):
        classifier = EmergencyClassifier()
        # Solid red civilian car (both roof and body are red)
        crop = np.zeros((80, 80, 3), dtype=np.uint8)
        crop[:, :] = [30, 30, 220]  # Red in BGR

        is_emer, conf, kind = classifier.check_roof_beacon_heuristic(crop)
        self.assertFalse(is_emer)
        self.assertEqual(kind, "civilian")

    def test_motorcycle_suppression(self):
        classifier = EmergencyClassifier()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[50:120, 50:90] = [255, 0, 0]  # Blue rider/bike
        tracks = [{
            "track_id": 99,
            "bbox": [50, 50, 90, 120],
            "class_name": "motorcycle",
            "centroid": (70, 85),
            "speed": 5.0,
            "stationary_frames": 0
        }]
        alerts = classifier.process_tracks(frame, tracks)
        self.assertEqual(len(alerts), 0)
        self.assertFalse(tracks[0]["is_emergency"])

    def test_incident_logging(self):
        log_path = "data/test_incidents.csv"
        logger = IncidentLogger(csv_path=log_path)
        inc = {
            "incident_id": "TEST-001",
            "type": "COLLISION_ACCIDENT",
            "severity": "CRITICAL",
            "vehicle_ids": [10, 11]
        }
        logger.log_incident(inc)
        recents = logger.get_recent_incidents()
        self.assertGreaterEqual(len(recents), 1)
        self.assertEqual(recents[0]["incident_id"], "TEST-001")
        if os.path.exists(log_path):
            os.remove(log_path)

    def test_sos_notifier(self):
        from src.sos_notifier import SOSNotifier
        # Test simulation mode (without credentials)
        notifier = SOSNotifier(recipient_email="test@example.com")
        incident = {
            "incident_id": "TEST-ACC-99",
            "timestamp": "2026-10-10 12:00:00",
            "severity": "CRITICAL",
            "vehicle_ids": [1, 2],
            "classes": ["car", "truck"]
        }
        loc = {
            "junction_name": "Test Junction",
            "latitude": 12.9716,
            "longitude": 77.5946,
            "is_simulated_gps": True
        }
        status = notifier.send_sos(incident, loc)
        self.assertIn("SIMULATED", status["status"])
        self.assertEqual(status["recipient_email"], "test@example.com")
        self.assertIn("Test Junction", status["location"])

        # Test configured credentials with simulated dispatch
        notifier_auth = SOSNotifier(
            recipient_email="test@example.com",
            smtp_user="sender@test.com",
            smtp_password="app_password"
        )
        self.assertEqual(notifier_auth.smtp_user, "sender@test.com")

    def test_emergency_route_advisor(self):
        from src.route_advisor import EmergencyRouteAdvisor
        advisor = EmergencyRouteAdvisor()
        # Test standby state
        res_standby = advisor.get_assistance([])
        self.assertFalse(res_standby["is_active"])
        self.assertEqual(res_standby["signal_phase"], "NORMAL_CYCLIC")

        # Test active emergency state
        emergencies = [{"track_id": 42, "subtype": "ambulance", "confidence": 0.88}]
        res_active = advisor.get_assistance(emergencies)
        self.assertTrue(res_active["is_active"])
        self.assertEqual(res_active["signal_phase"], "GREEN_WAVE_PRIORITY")
        self.assertIn("AMBULANCE #42", res_active["intersection_action"])
        self.assertGreater(len(res_active["corridor_waypoints"]), 0)


if __name__ == "__main__":
    unittest.main()

