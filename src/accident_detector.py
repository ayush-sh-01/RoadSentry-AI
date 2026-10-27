"""
RoadSentry AI - Accident Heuristic Engine Module
Detects potential road incidents using pairwise bounding-box IoU overlap, sudden deceleration, and trajectory anomalies.
"""

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import time
import numpy as np
import cv2


class AccidentDetector:
    """
    Rule-based incident heuristic engine.
    Heuristics:
      1. Pairwise Bounding-Box IoU Overlap (Spike > overlap_thresh).
      2. Sudden Deceleration (Pre-crash speed > min_speed, Post-event speed < stop_speed).
      3. Anomaly Persistence (Condition sustained across N consecutive frames to reject noise).
      4. Cooldown timer to prevent repetitive duplicate alerts for the same track.
    """

    def __init__(
        self,
        iou_thresh: float = 0.25,
        speed_drop_ratio: float = 0.70,
        min_pre_speed: float = 3.5,
        persistence_frames: int = 6,
        cooldown_frames: int = 60,
        enable_offroad_breach: bool = True,
        offroad_min_speed: float = 3.5,
        offroad_stop_speed: float = 1.2,
        offroad_persistence_frames: int = 5,
        offroad_cooldown_frames: int = 60,
        roi_polygon: Optional[List[Tuple[float, float]]] = None
    ):
        self.iou_thresh = iou_thresh
        self.speed_drop_ratio = speed_drop_ratio
        self.min_pre_speed = min_pre_speed
        self.persistence_frames = persistence_frames
        self.cooldown_frames = cooldown_frames

        # Off-road breach configuration
        self.enable_offroad_breach = enable_offroad_breach
        self.offroad_min_speed = offroad_min_speed
        self.offroad_stop_speed = offroad_stop_speed
        self.offroad_persistence_frames = offroad_persistence_frames
        self.offroad_cooldown_frames = offroad_cooldown_frames
        self.roi_normalized = roi_polygon

        # Pair anomaly tracker: (id1, id2) -> frame_count
        self.collision_persistence = defaultdict(int)
        self.cooldown_timers = defaultdict(int)

        # Single-vehicle off-road tracking: track_id -> frame_count
        self.offroad_persistence = defaultdict(int)
        self.single_cooldown_timers = defaultdict(int)
        # Track history of whether vehicle was previously inside road: track_id -> was_inside_bool
        self.track_was_inside = defaultdict(bool)

        # Track-specific speed history: track_id -> [s_t-4, ..., s_t]
        self.track_speed_history = defaultdict(list)

    @staticmethod
    def compute_iou(boxA: List[int], boxB: List[int]) -> float:
        """Computes Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = float(boxAArea + boxBArea - interArea)

        return interArea / unionArea if unionArea > 0 else 0.0

    def analyze_frame(
        self,
        tracked_objects: List[Dict[str, Any]],
        frame_idx: int = 0,
        frame_shape: Optional[Tuple[int, int, int]] = None,
        roi_polygon: Optional[List[Tuple[float, float]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyzes active tracks in current frame for incident heuristics:
        1. Pairwise vehicle-to-vehicle collision detection (IoU overlap + sudden deceleration).
        2. Single-vehicle off-road boundary breach & barrier crash detection.
        """
        # Decrement cooldown timers
        for key in list(self.cooldown_timers.keys()):
            if self.cooldown_timers[key] > 0:
                self.cooldown_timers[key] -= 1
            else:
                del self.cooldown_timers[key]

        for tid in list(self.single_cooldown_timers.keys()):
            if self.single_cooldown_timers[tid] > 0:
                self.single_cooldown_timers[tid] -= 1
            else:
                del self.single_cooldown_timers[tid]

        # Update speed histories
        for obj in tracked_objects:
            tid = obj["track_id"]
            sp = obj.get("speed", 0.0)
            self.track_speed_history[tid].append(sp)
            if len(self.track_speed_history[tid]) > 10:
                self.track_speed_history[tid].pop(0)

        active_incidents = []
        n = len(tracked_objects)

        # -------------------------------------------------------------
        # HEURISTIC 1: PAIRWISE VEHICLE-TO-VEHICLE COLLISION DETECTION
        # -------------------------------------------------------------
        for i in range(n):
            for j in range(i + 1, n):
                obj1 = tracked_objects[i]
                obj2 = tracked_objects[j]
                id1, id2 = sorted([obj1["track_id"], obj2["track_id"]])
                pair_key = (id1, id2)

                if pair_key in self.cooldown_timers:
                    continue

                # Check 1: Pairwise IoU overlap
                iou = self.compute_iou(obj1["bbox"], obj2["bbox"])

                # Check 2: Speed drop & relative stopping
                speeds1 = self.track_speed_history.get(id1, [])
                speeds2 = self.track_speed_history.get(id2, [])

                has_overlap = iou >= self.iou_thresh
                has_speed_drop = False

                if len(speeds1) >= 4 and len(speeds2) >= 4:
                    pre_speed1 = max(speeds1[:3])
                    curr_speed1 = speeds1[-1]
                    pre_speed2 = max(speeds2[:3])
                    curr_speed2 = speeds2[-1]

                    # Significant deceleration after being in motion
                    decel1 = (pre_speed1 > self.min_pre_speed and curr_speed1 < pre_speed1 * (1 - self.speed_drop_ratio))
                    decel2 = (pre_speed2 > self.min_pre_speed and curr_speed2 < pre_speed2 * (1 - self.speed_drop_ratio))
                    stationary_after_motion = (obj1.get("stationary_frames", 0) >= 3 or obj2.get("stationary_frames", 0) >= 3) and (pre_speed1 > 2.0 or pre_speed2 > 2.0)

                    if (decel1 or decel2 or stationary_after_motion):
                        has_speed_drop = True

                # Evaluate collision anomaly condition
                if has_overlap or (iou >= 0.15 and has_speed_drop):
                    self.collision_persistence[pair_key] += 1
                else:
                    self.collision_persistence[pair_key] = max(0, self.collision_persistence[pair_key] - 1)

                # Flag incident if sustained over persistence threshold
                if self.collision_persistence[pair_key] >= self.persistence_frames:
                    b1 = obj1["bbox"]
                    b2 = obj2["bbox"]
                    merged_box = [
                        min(b1[0], b2[0]),
                        min(b1[1], b2[1]),
                        max(b1[2], b2[2]),
                        max(b1[3], b2[3])
                    ]

                    incident = {
                        "incident_id": f"INC-{id1}-{id2}-{int(time.time())}",
                        "type": "COLLISION_ACCIDENT",
                        "severity": "CRITICAL",
                        "vehicle_ids": [id1, id2],
                        "classes": [obj1.get("class_name", "vehicle"), obj2.get("class_name", "vehicle")],
                        "bbox": merged_box,
                        "iou": round(float(iou), 3),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    active_incidents.append(incident)

                    # Trigger cooldown
                    self.cooldown_timers[pair_key] = self.cooldown_frames
                    self.collision_persistence[pair_key] = 0

        # -------------------------------------------------------------
        # HEURISTIC 2: SINGLE-VEHICLE OFF-ROAD BOUNDARY BREACH & BARRIER CRASH
        # -------------------------------------------------------------
        effective_roi = roi_polygon or self.roi_normalized
        if self.enable_offroad_breach and effective_roi and frame_shape:
            h, w = frame_shape[:2]
            try:
                roi_pts = np.array(
                    [[int(x * w), int(y * h)] for x, y in effective_roi],
                    dtype=np.int32
                )
            except Exception:
                roi_pts = None

            if roi_pts is not None and len(roi_pts) >= 3:
                for obj in tracked_objects:
                    tid = obj["track_id"]
                    if tid in self.single_cooldown_timers:
                        continue

                    cx, cy = obj["centroid"]
                    # Test if centroid is inside or on boundary of road ROI
                    dist_to_roi = cv2.pointPolygonTest(roi_pts, (float(cx), float(cy)), measureDist=True)
                    is_inside = dist_to_roi >= 0.0

                    speeds = self.track_speed_history.get(tid, [])
                    curr_speed = speeds[-1] if speeds else obj.get("speed", 0.0)

                    # Record if the vehicle was previously traveling inside the road
                    if is_inside and curr_speed >= self.offroad_min_speed:
                        self.track_was_inside[tid] = True

                    # Off-Road Crash Criteria:
                    # 1. Vehicle is now outside the road polygon (dist_to_roi < 0)
                    # 2. Was previously driving normally inside the road
                    # 3. Experienced abrupt halt / stoppage outside road (curr_speed < offroad_stop_speed)
                    # 4. Had prior kinetic motion (pre_speed >= offroad_min_speed)
                    if not is_inside and self.track_was_inside.get(tid, False):
                        pre_speeds = speeds[:-1] if len(speeds) > 1 else speeds
                        pre_speed = max(pre_speeds) if pre_speeds else curr_speed

                        has_stopped = curr_speed <= self.offroad_stop_speed or obj.get("stationary_frames", 0) >= 3
                        had_momentum = pre_speed >= self.offroad_min_speed

                        if has_stopped and had_momentum:
                            self.offroad_persistence[tid] += 1
                        else:
                            self.offroad_persistence[tid] = max(0, self.offroad_persistence[tid] - 1)
                    else:
                        self.offroad_persistence[tid] = max(0, self.offroad_persistence[tid] - 1)

                    # Trigger Off-Road Barrier Crash Incident
                    if self.offroad_persistence[tid] >= self.offroad_persistence_frames:
                        b = obj["bbox"]
                        incident = {
                            "incident_id": f"INC-OFFROAD-{tid}-{int(time.time())}",
                            "type": "OFFROAD_BARRIER_CRASH",
                            "severity": "CRITICAL",
                            "vehicle_ids": [tid],
                            "classes": [obj.get("class_name", "vehicle")],
                            "bbox": b,
                            "iou": 0.0,
                            "details": "Single-vehicle road departure & barrier/ditch collision",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        active_incidents.append(incident)

                        # Set cooldown and reset persistence
                        self.single_cooldown_timers[tid] = self.offroad_cooldown_frames
                        self.offroad_persistence[tid] = 0

        return active_incidents

    def draw_incidents(
        self,
        frame: np.ndarray,
        incidents: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Renders prominent visual alerting on incident bounding zones."""
        annotated = frame.copy()

        for inc in incidents:
            x1, y1, x2, y2 = inc["bbox"]
            # Bright Red alert bounding frame
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Warning banner header
            v_ids = inc.get("vehicle_ids", [])
            ids_str = ", ".join(f"#{i}" for i in v_ids)
            banner_text = f"CRITICAL: {inc.get('type', 'ACCIDENT')} (IDs {ids_str})"
            (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - 28)), (x1 + tw + 10, max(0, y1)), (0, 0, 255), -1)
            cv2.putText(
                annotated,
                banner_text,
                (x1 + 5, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return annotated

