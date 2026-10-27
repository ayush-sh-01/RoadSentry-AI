"""
Unit and Integration Tests for ThreadedStreamCapture
Verifies non-blocking reads, buffer management, thread safety, and auto-reconnect handling.
"""

import unittest
import time
import os
import numpy as np
import cv2

from src.stream_capture import ThreadedStreamCapture


class TestStreamCapture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_video = "data/test_videos/demo_traffic.mp4"
        if not os.path.exists(cls.test_video):
            from scripts.generate_sample_video import create_traffic_demo
            create_traffic_demo(cls.test_video)

    def test_threaded_capture_lifecycle(self):
        """Tests that the background thread starts, reads frames, and stops cleanly."""
        capture = ThreadedStreamCapture(source=self.test_video, reconnect_timeout=1.0)
        self.assertFalse(capture.is_alive())

        capture.start()
        self.assertTrue(capture.is_alive())

        # Wait a short duration for frames to be ingested
        for _ in range(20):
            time.sleep(0.1)
            has_frame, frame, tele = capture.read()
            if has_frame:
                break

        self.assertTrue(has_frame)
        self.assertIsNotNone(frame)
        self.assertTrue(tele["connected"])
        self.assertGreater(tele["total_frames"], 0)

        # Stop worker
        capture.stop()
        self.assertFalse(capture.is_alive())
        self.assertFalse(capture.is_connected())

    def test_non_blocking_read_on_empty(self):
        """Verifies read() does not hang or block even when no new frame has arrived."""
        capture = ThreadedStreamCapture(source="tests/dummy_non_existent.mp4", reconnect_timeout=0.2, max_reconnect_attempts=1)
        capture.start()

        # Non-blocking immediate check
        t0 = time.time()
        has_frame, frame, tele = capture.read()
        elapsed = time.time() - t0

        self.assertLess(elapsed, 0.1, "read() must be instantaneous and non-blocking")
        self.assertFalse(has_frame)
        self.assertIsNone(frame)

        capture.stop()

    def test_reconnection_trigger(self):
        """Verifies reconnect count increments when given an unreachable local file."""
        capture = ThreadedStreamCapture(
            source="tests/file_that_does_not_exist_404.mp4",
            reconnect_timeout=0.1,
            max_reconnect_attempts=2
        )
        capture.start()
        # Allow worker thread time to attempt open and record failure
        for _ in range(25):
            time.sleep(0.05)
            if capture.get_stats()["reconnects"] >= 1:
                break

        stats = capture.get_stats()
        self.assertGreaterEqual(stats["reconnects"], 1)
        self.assertFalse(stats["connected"])

        capture.stop()



if __name__ == "__main__":
    unittest.main()
