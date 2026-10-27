"""
RoadSentry AI - Threaded Live Stream Capture Worker
Ingests HLS (.m3u8), RTSP, HTTP, webcam, video files, and snapshot URLs in a background daemon thread.
Maintains a 1-slot latest-frame buffer to eliminate latency and buffer bloat, with auto-reconnection.
"""

import time
import threading
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

try:
    import requests
except ImportError:
    requests = None


class ThreadedStreamCapture:
    """
    High-performance, non-blocking video capture client.
    Runs frame decoding in a separate thread so inference is always fed the freshest frame.
    Includes auto-reconnect logic for intermittent network streams.
    """

    def __init__(
        self,
        source: Any,
        reconnect_timeout: float = 4.0,
        max_reconnect_attempts: int = 15,
        is_snapshot: bool = False,
        poll_fps: int = 15,
        name: str = "StreamCaptureWorker"
    ):
        """
        Args:
            source: URL string (HLS .m3u8, RTSP, HTTP, file path) or webcam index integer.
            reconnect_timeout: Seconds to wait between reconnection retries.
            max_reconnect_attempts: Max successive reconnection failures before aborting.
            is_snapshot: True if the source is a periodically updating JPEG/PNG image URL.
            poll_fps: Target polling frequency if is_snapshot is True.
            name: Worker thread identifier.
        """
        self.source = source
        self.reconnect_timeout = reconnect_timeout
        self.max_reconnect_attempts = max_reconnect_attempts
        self.is_snapshot = is_snapshot or (
            isinstance(source, str) and any(source.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"])
        )
        self.poll_interval = 1.0 / max(1, poll_fps)
        self.name = name

        # Thread synchronization
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Frame state
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_timestamp: float = 0.0
        self._frame_count: int = 0
        self._drop_count: int = 0
        self._is_connected: bool = False
        self._reconnect_count: int = 0
        self._last_error: str = ""

        # Performance metrics
        self._fps_meter_time: float = time.time()
        self._fps_meter_frames: int = 0
        self._current_ingest_fps: float = 0.0

        # Underlying capture
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> "ThreadedStreamCapture":
        """Starts the background capture thread."""
        if self._running:
            return self

        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, name=self.name, daemon=True)
        self._thread.start()
        return self

    def _open_stream(self) -> bool:
        """Initializes the underlying cv2.VideoCapture."""
        if self.is_snapshot:
            self._is_connected = True
            return True

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

        # Optimize for low-latency network streams
        cap = cv2.VideoCapture(self.source)
        try:
            # Drop OpenCV internal buffer to minimum
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if cap.isOpened():
            self._cap = cap
            self._is_connected = True
            self._last_error = ""
            return True
        else:
            self._cap = None
            self._is_connected = False
            self._last_error = f"Failed to open video source: {self.source}"
            return False

    def _fetch_snapshot(self) -> Optional[np.ndarray]:
        """Fetches and decodes an image from a static HTTP snapshot endpoint."""
        if requests is None or not isinstance(self.source, str):
            return None
        try:
            resp = requests.get(self.source, timeout=3.0)
            if resp.status_code == 200:
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return img
        except Exception as e:
            self._last_error = str(e)
        return None

    def _capture_worker(self):
        """Continuous background thread worker."""
        consecutive_failures = 0

        while self._running:
            if not self._is_connected or (not self.is_snapshot and (self._cap is None or not self._cap.isOpened())):
                success = self._open_stream()
                if not success:
                    consecutive_failures += 1
                    with self._lock:
                        self._reconnect_count += 1
                    if consecutive_failures > self.max_reconnect_attempts:
                        self._last_error = f"Exceeded max reconnect attempts ({self.max_reconnect_attempts})"
                        time.sleep(self.reconnect_timeout)
                        consecutive_failures = 0
                    else:
                        time.sleep(min(self.reconnect_timeout, 0.5))
                    continue
                else:
                    consecutive_failures = 0

            # Mode 1: Snapshot API Polling
            if self.is_snapshot:
                frame = self._fetch_snapshot()
                if frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                        self._frame_timestamp = time.time()
                        self._frame_count += 1
                    self._update_fps()
                else:
                    self._is_connected = False
                time.sleep(self.poll_interval)
                continue

            # Mode 2: Continuous Video Stream (HLS / RTSP / Video file / Webcam)
            try:
                ret, frame = self._cap.read()
                if not ret or frame is None or frame.size == 0:
                    # Stream interrupted or ended
                    self._is_connected = False
                    consecutive_failures += 1
                    self._reconnect_count += 1
                    time.sleep(0.5)
                    continue

                consecutive_failures = 0
                now = time.time()

                with self._lock:
                    if self._latest_frame is not None:
                        self._drop_count += 1
                    self._latest_frame = frame
                    self._frame_timestamp = now
                    self._frame_count += 1

                self._update_fps()

            except Exception as e:
                self._is_connected = False
                self._last_error = str(e)
                time.sleep(1.0)

        # Cleanup on thread exit
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_connected = False

    def _update_fps(self):
        """Calculates rolling capture FPS."""
        self._fps_meter_frames += 1
        elapsed = time.time() - self._fps_meter_time
        if elapsed >= 1.0:
            self._current_ingest_fps = self._fps_meter_frames / elapsed
            self._fps_meter_frames = 0
            self._fps_meter_time = time.time()

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Reads the most recent frame without blocking.
        Returns:
            Tuple[has_frame (bool), frame (ndarray or None), telemetry (dict)]
        """
        with self._lock:
            frame = self._latest_frame
            # Clear stored reference so caller knows if new frame arrived
            self._latest_frame = None
            t_stamp = self._frame_timestamp

        has_frame = frame is not None
        telemetry = {
            "connected": self._is_connected,
            "timestamp": t_stamp,
            "total_frames": self._frame_count,
            "dropped_frames": self._drop_count,
            "ingest_fps": self._current_ingest_fps,
            "reconnect_count": self._reconnect_count,
            "last_error": self._last_error
        }
        return has_frame, frame, telemetry

    def is_alive(self) -> bool:
        """Returns True if the worker thread is actively running."""
        return self._running and self._thread is not None and self._thread.is_alive()

    def is_connected(self) -> bool:
        """Returns True if the video capture is healthy and receiving frames."""
        return self._is_connected

    def get_stats(self) -> Dict[str, Any]:
        """Returns capture health and performance metrics."""
        with self._lock:
            return {
                "source": str(self.source),
                "connected": self._is_connected,
                "ingest_fps": round(self._current_ingest_fps, 1),
                "total_frames": self._frame_count,
                "dropped_frames": self._drop_count,
                "reconnects": self._reconnect_count,
                "last_error": self._last_error
            }

    def stop(self):
        """Signals the worker thread to stop and releases resources."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_connected = False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
