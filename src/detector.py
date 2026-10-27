"""
RoadSentry AI - Vehicle Detector Module
Implements YOLOv8 detection for vehicles (Car, Bus, Truck, Motorcycle).
"""

from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from ultralytics import YOLO


from src.classifier_refiner import VehicleClassifierRefiner


class VehicleDetector:
    """
    Wraps YOLOv8 model for road vehicle detection and classification.
    Target vehicle classes (COCO):
      1: bicycle (mapped to motorcycle/two-wheeler)
      2: car
      3: motorcycle
      5: bus
      7: truck
    """

    CLASS_NAMES = {
        1: "motorcycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    CLASS_COLORS = {
        "car": (0, 215, 255),         # Gold/Amber
        "motorcycle": (255, 128, 0),  # Blue
        "bus": (255, 0, 128),         # Purple/Magenta
        "truck": (0, 165, 255),       # Orange
        "emergency": (0, 0, 255)      # Red
    }

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        target_classes: Optional[List[int]] = None,
        device: str = "cpu"
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.target_classes = target_classes if target_classes is not None else [1, 2, 3, 5, 7]
        
        # Resolve 'auto' device safely for ultralytics
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.refiner = VehicleClassifierRefiner()
        
        # Load YOLO model (will download yolov8n.pt automatically on first run)
        self.model = YOLO(model_name)


    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run vehicle detection on a single RGB/BGR frame.

        Returns a list of dicts:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "class_name": str,
                "centroid": (cx, cy)
            },
            ...
        ]
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.target_classes,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results:
            return detections

        r = results[0]
        boxes = r.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                raw_class_name = self.CLASS_NAMES.get(cls_id, "vehicle")

                # Refine class using geometry, aspect ratio and perspective scale
                refined_class, refined_conf = self.refiner.refine_single_box(
                    bbox=[x1, y1, x2, y2],
                    raw_class_name=raw_class_name,
                    confidence=float(conf),
                    frame_shape=frame.shape
                )

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(refined_conf),
                    "class_id": int(cls_id),
                    "class_name": refined_class,
                    "raw_class_name": raw_class_name,
                    "centroid": (cx, cy)
                })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes and class labels on frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = self.CLASS_COLORS.get(det["class_name"], (0, 255, 0))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label banner
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w + 4, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        return annotated
