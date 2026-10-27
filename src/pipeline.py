"""
RoadSentry AI - Master Analytics Pipeline
Connects Detection, Tracking, Density Estimation, Heatmaps, Accident Heuristics, and Emergency Classification into a cohesive engine.
"""

from typing import Dict, Any, Optional, List, Tuple
import time
import numpy as np
import cv2

from src.tracker import VehicleTracker
from src.density import DensityEstimator
from src.heatmap import HeatmapAccumulator
from src.accident_detector import AccidentDetector
from src.emergency_classifier import EmergencyClassifier
from src.logger import IncidentLogger
from src.sos_notifier import SOSNotifier
from src.route_advisor import EmergencyRouteAdvisor


class RoadSentryPipeline:
    """
    Main real-time vision analytics engine.
    Integrates Traffic Monitoring, Live Heatmaps, Emergency Route Assistance, and Accident SOS Alerts.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.20,
        iou_thresh: float = 0.65,
        low_density_thresh: int = 2,
        medium_density_thresh: int = 4,
        roi_polygon: Optional[List[Tuple[float, float]]] = None,
        heatmap_decay: float = 0.96,
        accident_iou_thresh: float = 0.25,
        enable_emergency_detection: bool = True,
        location_data: Optional[Dict[str, Any]] = None,
        sos_config: Optional[Dict[str, Any]] = None,
        route_config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        **kwargs
    ):
        # Location & GIS metadata
        self.location_data = location_data or {
            "camera_id": "CAM-01",
            "junction_name": "Central Ring Road - Hospital Junction",
            "latitude": 12.9716,
            "longitude": 77.5946,
            "is_simulated_gps": True
        }

        # 1. Tracker (YOLOv8 + ByteTrack)
        self.tracker = VehicleTracker(
            model_name=model_name,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            device=device
        )

        # 2. Density & ROI
        self.density = DensityEstimator(
            low_thresh=low_density_thresh,
            medium_thresh=medium_density_thresh,
            roi_polygon=roi_polygon
        )

        # 3. Heatmap
        self.heatmap = HeatmapAccumulator(decay_factor=heatmap_decay)

        # 4. Accident Detection (Pairwise Collisions & Off-road Barrier Crashes)
        accident_cfg = kwargs.get("accident_config", {})
        acc_detector_kwargs = {
            "iou_thresh": accident_iou_thresh,
            "speed_drop_ratio": float(accident_cfg.get("speed_drop_ratio", 0.70)),
            "min_pre_speed": float(accident_cfg.get("min_pre_speed", 3.5)),
            "persistence_frames": int(accident_cfg.get("persistence_frames", 6)),
            "cooldown_frames": int(accident_cfg.get("cooldown_frames", 60)),
            "enable_offroad_breach": bool(accident_cfg.get("enable_offroad_breach", True)),
            "offroad_min_speed": float(accident_cfg.get("offroad_min_speed", 3.5)),
            "offroad_stop_speed": float(accident_cfg.get("offroad_stop_speed", 1.2)),
            "offroad_persistence_frames": int(accident_cfg.get("offroad_persistence_frames", 5)),
            "offroad_cooldown_frames": int(accident_cfg.get("offroad_cooldown_frames", 60)),
            "roi_polygon": roi_polygon
        }
        self.accident_detector = AccidentDetector(**acc_detector_kwargs)
        self.roi_polygon = roi_polygon

        # 5. Emergency Classifier
        self.emergency_classifier = EmergencyClassifier(
            enable_beacon_heuristic=enable_emergency_detection
        )

        # 6. Incident Logger
        self.logger = IncidentLogger()
        self.logged_emergency_ids = set()

        # 7. SOS Notifier (Feature 4)
        self.sos_notifier = SOSNotifier(**(sos_config or {}))

        # 8. Route & Intersection Advisor (Feature 3)
        self.route_advisor = EmergencyRouteAdvisor(**(route_config or {}))

        self.frame_count = 0
        self.fps = 0.0
        self._last_time = time.time()

    def process_frame(
        self,
        frame: np.ndarray,
        show_tracks: bool = True,
        show_trails: bool = True,
        show_roi: bool = True,
        show_heatmap: bool = False,
        show_accidents: bool = True,
        show_emergency: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single video frame and produces annotated output and telemetry data.
        """
        self.frame_count += 1
        curr_time = time.time()
        elapsed = curr_time - self._last_time
        if elapsed > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / elapsed)
        self._last_time = curr_time

        # Step 1: Vehicle Tracking & Classification
        tracks = self.tracker.track(frame)
        active_tids = [t["track_id"] for t in tracks]

        # Feature 1: Category-wise vehicle counts
        counts_by_class = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }
        for t in tracks:
            cls = t.get("class_name", "car")
            if cls in counts_by_class:
                counts_by_class[cls] += 1
            elif cls == "bicycle":
                counts_by_class["motorcycle"] += 1

        # Step 2: Emergency Vehicle Classification
        emergencies = []
        if show_emergency:
            emergencies = self.emergency_classifier.process_tracks(frame, tracks)
            self.emergency_classifier.clear_inactive_tracks(active_tids)
            for em in emergencies:
                # Deduplicate incident logging: log once per emergency vehicle track
                if em["track_id"] not in self.logged_emergency_ids:
                    self.logged_emergency_ids.add(em["track_id"])
                    self.logger.log_incident({
                        "incident_id": em["alert_id"],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "severity": "HIGH",
                        "type": f"PRIORITY_{em['subtype'].upper()}",
                        "vehicle_ids": [em["track_id"]],
                        "bbox": em["bbox"]
                    }, frame)

        # Feature 3: Route & Intersection Assistance
        route_assistance = self.route_advisor.get_assistance(emergencies, self.location_data)

        # Step 3: Density Estimation (Feature 1)
        density_data = self.density.estimate(frame.shape, tracks)

        # Step 4: Heatmap Accumulation (Feature 2)
        self.heatmap.update(frame.shape, tracks)

        # Step 5: Accident Anomaly Detection & SOS Dispatch (Feature 4)
        accidents = self.accident_detector.analyze_frame(
            tracked_objects=tracks,
            frame_idx=self.frame_count,
            frame_shape=frame.shape,
            roi_polygon=self.roi_polygon
        )
        for acc in accidents:
            self.logger.log_incident(acc, frame)
            # Dispatch real/simulated SOS email notification
            self.sos_notifier.send_sos(acc, self.location_data)

        # Step 6: Render Visual Overlays
        annotated = frame.copy()

        # Heatmap overlay if requested
        if show_heatmap:
            annotated = self.heatmap.blend_overlay(annotated, alpha=0.45)

        # ROI & Density HUD
        if show_roi:
            annotated = self.density.draw_roi_and_density(annotated, density_data)

        # Tracks & Trails
        if show_tracks:
            annotated = self.tracker.draw_tracks(annotated, tracks, draw_trails=show_trails)

        # Emergency Badges
        if show_emergency and emergencies:
            annotated = self.emergency_classifier.draw_emergency_badges(annotated, emergencies)

        # Accidents Alert Box
        if show_accidents and accidents:
            annotated = self.accident_detector.draw_incidents(annotated, accidents)

        # Draw FPS watermark
        cv2.putText(
            annotated,
            f"RoadSentry AI | FPS: {self.fps:.1f}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Safe heatmap preview and stats extraction with fallbacks
        if hasattr(self.heatmap, "get_annotated_heatmap_view"):
            heatmap_preview = self.heatmap.get_annotated_heatmap_view(frame)
        elif hasattr(self.heatmap, "blend_overlay"):
            heatmap_preview = self.heatmap.blend_overlay(frame)
        else:
            heatmap_preview = frame

        if hasattr(self.heatmap, "get_hotspot_summary"):
            heatmap_stats = self.heatmap.get_hotspot_summary()
        else:
            heatmap_stats = {"peak_intensity": 0, "bottleneck_level": "CLEAR", "bottleneck_desc": "Road is clear"}

        heatmap_raw = self.heatmap.get_colored_heatmap() if hasattr(self.heatmap, "get_colored_heatmap") else frame

        telemetry = {
            "frame_idx": self.frame_count,
            "fps": round(self.fps, 1),
            "vehicle_count": len(tracks),
            "counts_by_class": counts_by_class,
            "roi_count": density_data["roi_count"],
            "density_level": density_data["density_level"],
            "occupancy_ratio": density_data.get("occupancy_ratio", 0.0),
            "avg_speed": density_data.get("avg_speed", 5.0),
            "accidents": accidents,
            "emergencies": emergencies,
            "tracks": tracks,
            "route_assistance": route_assistance,
            "sos_status": self.sos_notifier.last_status,
            "location_data": self.location_data,
            "heatmap_preview": heatmap_preview,
            "heatmap_raw": heatmap_raw,
            "heatmap_stats": heatmap_stats
        }

        return annotated, telemetry
