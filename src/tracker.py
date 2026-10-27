"""
RoadSentry AI - Multi-Object Tracker Module
Wraps Ultralytics ByteTrack for persistent vehicle ID assignment and trajectory/velocity estimation.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
import numpy as np
import cv2
from ultralytics import YOLO


from src.classifier_refiner import VehicleClassifierRefiner


class VehicleTracker:
    """
    Multi-object vehicle tracker using ByteTrack.
    Maintains persistent IDs, trajectories, instantaneous velocities, and stabilized classifications.
    """

    CLASS_NAMES = {
        1: "motorcycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_type: str = "bytetrack.yaml",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        target_classes: Optional[List[int]] = None,
        max_trajectory_len: int = 30,
        device: str = "cpu"
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.target_classes = target_classes if target_classes is not None else [1, 2, 3, 5, 7]
        self.tracker_type = tracker_type
        self.device = device
        self.max_trajectory_len = max_trajectory_len

        # Persistent history across frames
        self.trajectories = defaultdict(lambda: deque(maxlen=self.max_trajectory_len))
        self.speeds = defaultdict(float)  # pixel velocity per frame
        self.stationary_counters = defaultdict(int)
        self.refiner = VehicleClassifierRefiner()

        self.model = YOLO(model_name)

    def track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Track vehicles across frames.
        Returns list of active tracked vehicle dictionaries.
        """
        results = self.model.track(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.target_classes,
            tracker=self.tracker_type,
            persist=True,
            device=self.device,
            verbose=False
        )

        tracked_objects = []
        if not results:
            return tracked_objects

        r = results[0]
        boxes = r.boxes

        if boxes is not None and boxes.id is not None and len(boxes.id) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for box, track_id, conf, cls_id in zip(xyxy, track_ids, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                raw_class_name = self.CLASS_NAMES.get(cls_id, "vehicle")

                # Stabilize and refine classification using geometry & temporal track voting
                stabilized_class, smoothed_conf = self.refiner.update_track_class(
                    track_id=int(track_id),
                    bbox=[x1, y1, x2, y2],
                    raw_class_name=raw_class_name,
                    confidence=float(conf),
                    frame_shape=frame.shape
                )

                # Update trajectory history
                traj = self.trajectories[track_id]
                traj.append((cx, cy))

                # Compute instantaneous velocity (displacement over last 5 frames)
                if len(traj) >= 5:
                    dx = traj[-1][0] - traj[-5][0]
                    dy = traj[-1][1] - traj[-5][1]
                    speed = np.sqrt(dx ** 2 + dy ** 2) / 4.0
                    self.speeds[track_id] = float(speed)
                    if speed < 1.0:
                        self.stationary_counters[track_id] += 1
                    else:
                        self.stationary_counters[track_id] = 0
                else:
                    speed = self.speeds[track_id]

                tracked_objects.append({
                    "track_id": int(track_id),
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(smoothed_conf),
                    "class_id": int(cls_id),
                    "class_name": stabilized_class,
                    "raw_class_name": raw_class_name,
                    "centroid": (cx, cy),
                    "speed": float(speed),
                    "trajectory": list(traj),
                    "stationary_frames": int(self.stationary_counters[track_id])
                })

            # Clean up old tracks
            self.refiner.clear_inactive_tracks(list(track_ids))

        return tracked_objects

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        draw_trails: bool = True
    ) -> np.ndarray:
        """Render vehicle bounding boxes, IDs, speed telemetry, and movement trails."""
        annotated = frame.copy()

        for obj in tracks:
            track_id = obj["track_id"]
            x1, y1, x2, y2 = obj["bbox"]
            cls_name = obj["class_name"]
            speed = obj["speed"]
            traj = obj["trajectory"]

            # Draw trajectory path
            if draw_trails and len(traj) > 1:
                pts = np.array(traj, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

            # Box color based on speed (Cyan if moving, Orange if slow/stopped)
            box_color = (0, 200, 255) if speed >= 2.0 else (0, 140, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Telemetry label
            label = f"#{track_id} {cls_name} ({speed:.1f}px/f)"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - 18), (x1 + w + 4, y1), (20, 20, 20), -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return annotated
