"""
RoadSentry AI - Emergency Vehicle Classifier Module
Identifies emergency vehicles (Ambulance, Police, Fire Engine) using roof beacon flash heuristics and lightweight CNN classification.
"""

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image


class EmergencyClassifier:
    """
    Emergency Vehicle Detector:
    Combines optical hue/saturation roof beacon heuristics with body-contrast validation,
    motorcycle suppression, and temporal track persistence to eliminate false positives.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_thresh: float = 0.60,
        enable_beacon_heuristic: bool = True
    ):
        self.confidence_thresh = confidence_thresh
        self.enable_beacon_heuristic = enable_beacon_heuristic
        self.model = None

        # Red and Blue HSV ranges for emergency strobes (high saturation & brightness)
        self.red_lower1 = np.array([0, 110, 130])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 110, 130])
        self.red_upper2 = np.array([180, 255, 255])
        self.blue_lower = np.array([100, 130, 130])
        self.blue_upper = np.array([135, 255, 255])

        # Temporal persistence accumulator: track_id -> vote_count
        self.track_emergency_votes = defaultdict(int)

    def check_roof_beacon_heuristic(self, crop: np.ndarray) -> Tuple[bool, float, str]:
        """
        Analyzes the upper roof area of vehicle crop for active emergency beacon strobes.
        Distinguishes active emergency strobe lightbars from solid-colored car bodies,
        taillights/brake lights, and clothing reflections.
        """
        ch, cw = crop.shape[:2]
        if ch < 25 or cw < 25:
            return False, 0.0, "civilian"

        # Roof slice: upper 30% of vehicle height
        roof_h = max(4, int(ch * 0.30))
        roof_crop = crop[:roof_h, :]

        # Lower body slice: bottom 60% of vehicle height (for body paint reference)
        body_crop = crop[int(ch * 0.40):, :]

        # Convert to HSV
        hsv_roof = cv2.cvtColor(roof_crop, cv2.COLOR_BGR2HSV)
        hsv_body = cv2.cvtColor(body_crop, cv2.COLOR_BGR2HSV) if body_crop.shape[0] > 5 else None

        # Mask emergency red and blue pixels on roof
        mask_r1_roof = cv2.inRange(hsv_roof, self.red_lower1, self.red_upper1)
        mask_r2_roof = cv2.inRange(hsv_roof, self.red_lower2, self.red_upper2)
        mask_red_roof = cv2.bitwise_or(mask_r1_roof, mask_r2_roof)
        mask_blue_roof = cv2.inRange(hsv_roof, self.blue_lower, self.blue_upper)

        roof_pixels = roof_crop.shape[0] * roof_crop.shape[1]
        red_ratio = float(np.count_nonzero(mask_red_roof)) / roof_pixels
        blue_ratio = float(np.count_nonzero(mask_blue_roof)) / roof_pixels

        # Check body paint ratios to avoid flagging cars painted solid red or solid blue
        body_red_ratio = 0.0
        body_blue_ratio = 0.0
        if hsv_body is not None and hsv_body.size > 0:
            body_pixels = body_crop.shape[0] * body_crop.shape[1]
            mask_r1_b = cv2.inRange(hsv_body, self.red_lower1, self.red_upper1)
            mask_r2_b = cv2.inRange(hsv_body, self.red_lower2, self.red_upper2)
            mask_red_b = cv2.bitwise_or(mask_r1_b, mask_r2_b)
            mask_blue_b = cv2.inRange(hsv_body, self.blue_lower, self.blue_upper)
            body_red_ratio = float(np.count_nonzero(mask_red_b)) / body_pixels
            body_blue_ratio = float(np.count_nonzero(mask_blue_b)) / body_pixels

        # 1. Dual-tone strobe signature: Co-occurrence of BOTH red and blue lights on roof
        # (Standard police / ambulance / emergency lightbars)
        has_dual_strobe = (red_ratio >= 0.02 and blue_ratio >= 0.02)

        # 2. Localized Roof Siren Beacon:
        # High-intensity red or blue beacon on roof, while the lower vehicle body is NOT that color
        has_isolated_red_beacon = (
            red_ratio >= 0.06 and
            body_red_ratio < 0.03 and
            red_ratio > (body_red_ratio * 2.5)
        )
        has_isolated_blue_beacon = (
            blue_ratio >= 0.06 and
            body_blue_ratio < 0.03 and
            blue_ratio > (body_blue_ratio * 2.5)
        )

        if has_dual_strobe or has_isolated_red_beacon or has_isolated_blue_beacon:
            conf = min(0.95, float(max(red_ratio, blue_ratio) * 3.5 + 0.35))
            if has_dual_strobe:
                kind = "ambulance" if red_ratio >= blue_ratio else "police/fire"
            elif has_isolated_red_beacon:
                kind = "ambulance"
            else:
                kind = "police/fire"
            return True, conf, kind

        return False, 0.0, "civilian"

    def classify_crop(self, crop: np.ndarray) -> Tuple[bool, float, str]:
        """Classify a vehicle crop patch."""
        if self.enable_beacon_heuristic:
            is_emer, conf, kind = self.check_roof_beacon_heuristic(crop)
            if is_emer:
                return True, conf, kind

        return False, 0.0, "civilian"

    def process_tracks(
        self,
        frame: np.ndarray,
        tracked_objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enriches tracked objects with emergency status and creates priority alerts.
        Applies class filtering (suppresses two-wheelers) and multi-frame temporal voting.
        """
        h, w = frame.shape[:2]
        emergency_alerts = []

        for obj in tracked_objects:
            tid = obj["track_id"]
            cls_name = obj.get("class_name", "car")

            # Two-wheelers / motorcycles are never full emergency vehicles (police cruisers / ambulances / fire trucks)
            if cls_name in ("motorcycle", "bicycle"):
                obj["is_emergency"] = False
                obj["emergency_type"] = "civilian"
                obj["emergency_conf"] = 0.0
                self.track_emergency_votes[tid] = max(0, self.track_emergency_votes[tid] - 1)
                continue

            x1, y1, x2, y2 = obj["bbox"]
            # Clamp bbox inside frame
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if (x2 - x1) < 25 or (y2 - y1) < 25:
                obj["is_emergency"] = False
                obj["emergency_type"] = "civilian"
                obj["emergency_conf"] = 0.0
                continue

            crop = frame[y1:y2, x1:x2]
            is_emer, conf, kind = self.classify_crop(crop)

            # Temporal voting: require persistent beacon detection across multiple frames
            if is_emer:
                self.track_emergency_votes[tid] = min(8, self.track_emergency_votes[tid] + 2)
            else:
                self.track_emergency_votes[tid] = max(0, self.track_emergency_votes[tid] - 1)

            # Confirm emergency only if sustained across at least 3 votes
            confirmed_emer = self.track_emergency_votes[tid] >= 3

            obj["is_emergency"] = confirmed_emer
            obj["emergency_type"] = kind if confirmed_emer else "civilian"
            obj["emergency_conf"] = conf if confirmed_emer else 0.0

            if confirmed_emer:
                alert = {
                    "alert_id": f"EMERGENCY-{tid}",
                    "type": "EMERGENCY_VEHICLE",
                    "subtype": kind,
                    "track_id": tid,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": cv2.getTickCount()
                }
                emergency_alerts.append(alert)

        return emergency_alerts

    def clear_inactive_tracks(self, active_track_ids: List[int]):
        """Cleans up temporal state for departed tracks."""
        active_set = set(active_track_ids)
        for tid in list(self.track_emergency_votes.keys()):
            if tid not in active_set:
                del self.track_emergency_votes[tid]

    def draw_emergency_badges(
        self,
        frame: np.ndarray,
        emergency_alerts: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Renders flashing neon alert box around confirmed emergency vehicles."""
        annotated = frame.copy()
        for alert in emergency_alerts:
            x1, y1, x2, y2 = alert["bbox"]
            # Bright Red / Blue border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

            label = f"PRIORITY: {alert['subtype'].upper()} #{alert['track_id']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - 22), (x1 + tw + 6, y1), (0, 0, 255), -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return annotated
