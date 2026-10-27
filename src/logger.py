"""
RoadSentry AI - Incident Logger Module
Maintains incident event logs, exports to CSV/JSON, and captures incident snapshots.
"""

from typing import List, Dict, Any, Optional
import os
import json
import time
import pandas as pd
import cv2


class IncidentLogger:
    """
    Persists incident telemetry and manages alert logs.
    """

    def __init__(
        self,
        csv_path: str = "data/incident_logs.csv",
        snapshot_dir: str = "data/snapshots"
    ):
        self.csv_path = csv_path
        self.snapshot_dir = snapshot_dir
        self.incident_history: List[Dict[str, Any]] = []

        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Initialize CSV file if not existing
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                "incident_id", "timestamp", "severity", "type",
                "vehicle_ids", "details", "snapshot_path"
            ])
            df.to_csv(self.csv_path, index=False)

    def log_incident(
        self,
        incident: Dict[str, Any],
        frame: Optional[cv2.Mat] = None
    ) -> Dict[str, Any]:
        """
        Record a new incident, optionally capturing a frame crop snapshot.
        """
        timestamp = incident.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        inc_id = incident.get("incident_id", f"INC-{int(time.time() * 1000)}")
        severity = incident.get("severity", "MEDIUM")
        inc_type = incident.get("type", "ANOMALY")
        v_ids = str(incident.get("vehicle_ids", []))

        # Save snapshot image if frame provided
        snapshot_file = ""
        if frame is not None and "bbox" in incident:
            x1, y1, x2, y2 = incident["bbox"]
            h, w = frame.shape[:2]
            # Add padding around box
            pad = 20
            px1 = max(0, x1 - pad)
            py1 = max(0, y1 - pad)
            px2 = min(w, x2 + pad)
            py2 = min(h, y2 + pad)
            crop = frame[py1:py2, px1:px2]

            if crop.size > 0:
                snapshot_file = os.path.join(self.snapshot_dir, f"{inc_id}.jpg")
                cv2.imwrite(snapshot_file, crop)

        record = {
            "incident_id": inc_id,
            "timestamp": timestamp,
            "severity": severity,
            "type": inc_type,
            "vehicle_ids": v_ids,
            "details": f"IoU: {incident.get('iou', 0.0)} | Classes: {incident.get('classes', [])}",
            "snapshot_path": snapshot_file
        }

        self.incident_history.insert(0, record)

        # Append to CSV disk storage
        df_row = pd.DataFrame([record])
        df_row.to_csv(self.csv_path, mode="a", header=False, index=False)

        return record

    def get_recent_incidents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent recorded incidents."""
        return self.incident_history[:limit]

    def export_json(self) -> str:
        """Export history as JSON string."""
        return json.dumps(self.incident_history, indent=2)

    def get_dataframe(self) -> pd.DataFrame:
        """Return incident history as pandas DataFrame."""
        if os.path.exists(self.csv_path):
            try:
                return pd.read_csv(self.csv_path)
            except Exception:
                pass
        return pd.DataFrame(self.incident_history)
