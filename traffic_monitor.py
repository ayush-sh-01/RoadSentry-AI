"""
RoadSentry AI - Standalone Live Traffic Camera Monitor
Runs low-latency vehicle detection, multi-class counting, and on-screen HUD
from live HLS streams, RTSP links, snapshots, or video files.
"""

import sys
import time
import argparse
from typing import Dict, List, Any
import numpy as np
import cv2

from config.config import (
    STREAM_URL,
    DEFAULT_MODEL,
    CONF_THRESH,
    IOU_THRESH,
    DEVICE,
    TARGET_CLASSES
)
from src.stream_capture import ThreadedStreamCapture
from src.detector import VehicleDetector
from src.density import DensityEstimator


def draw_hud(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    counts: Dict[str, int],
    density_label: str,
    fps: float,
    ingest_fps: float,
    source_name: str
) -> np.ndarray:
    """
    Renders professional telemetry HUD over frame:
    - Bounding boxes with category colors
    - Top telemetry bar with FPS, density, and category pills
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Colors (BGR)
    COLOR_BG = (15, 23, 42)          # Deep slate
    COLOR_BORDER = (51, 65, 85)      # Slate border
    COLOR_TEXT = (248, 250, 252)     # Off-white
    COLOR_CYAN = (255, 229, 0)       # Cyan in BGR
    COLOR_AMBER = (0, 183, 255)      # Amber in BGR
    COLOR_GREEN = (118, 230, 0)      # Green in BGR

    # Draw Bounding Boxes
    for det in detections:
        box = det.get("bbox", det.get("box", [0, 0, 0, 0]))
        cls_name = det.get("class_name", "vehicle")
        conf = det.get("confidence", 0.0)

        x1, y1, x2, y2 = map(int, box)
        color = VehicleDetector.CLASS_COLORS.get(cls_name, (0, 255, 0))

        # Main box with corner accents
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label tag
        label = f"{cls_name.upper()} {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 5, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    # Top HUD Bar Overlay
    hud_h = 70
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.line(canvas, (0, hud_h), (w, hud_h), COLOR_BORDER, 1)

    # Left: Title and Source
    cv2.putText(canvas, "ROADSENTRY AI - LIVE TRAFFIC MONITOR", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CYAN, 2, cv2.LINE_AA)
    src_display = source_name if len(source_name) < 42 else source_name[:39] + "..."
    cv2.putText(canvas, f"SRC: {src_display}", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)

    # Center: Category Counts
    cat_x = int(w * 0.36)
    categories = [
        ("CARS", counts.get("car", 0), (0, 215, 255)),
        ("BIKES", counts.get("motorcycle", 0), (255, 128, 0)),
        ("BUSES", counts.get("bus", 0), (255, 0, 128)),
        ("TRUCKS", counts.get("truck", 0), (0, 165, 255)),
    ]

    for i, (cat_label, cnt, c_color) in enumerate(categories):
        bx = cat_x + (i * 90)
        cv2.rectangle(canvas, (bx, 14), (bx + 82, 54), COLOR_BORDER, 1)
        cv2.putText(canvas, cat_label, (bx + 8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(cnt), (bx + 8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_color, 2, cv2.LINE_AA)

    # Right: Density & FPS
    right_x = max(cat_x + 380, w - 240)
    density_color = COLOR_GREEN if density_label == "LOW" else (COLOR_AMBER if density_label == "MEDIUM" else (0, 0, 255))
    cv2.putText(canvas, f"DENSITY: {density_label}", (right_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, density_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"PROC: {fps:.1f} FPS | CAM: {ingest_fps:.1f} FPS", (right_x, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)

    return canvas


def run_traffic_monitor(stream_url: str, model_path: str = DEFAULT_MODEL, headless: bool = False):
    """Executes live vehicle detection loop on video stream."""
    print("=" * 65)
    print("🚦 RoadSentry AI - Live Traffic Camera Stream Monitor")
    print(f"📡 Ingest Stream URL : {stream_url}")
    print(f"🧠 Detection Model    : {model_path}")
    print(f"🎯 Target Classes     : Cars, Motorcycles, Buses, Trucks")
    print("=" * 65)

    # 1. Initialize Ingestion Worker
    capture = ThreadedStreamCapture(source=stream_url).start()

    # 2. Initialize Model & Analytics
    detector = VehicleDetector(
        model_name=model_path,
        conf_thresh=CONF_THRESH,
        iou_thresh=IOU_THRESH,
        target_classes=TARGET_CLASSES,
        device=DEVICE
    )
    density_estimator = DensityEstimator(low_thresh=2, medium_thresh=4)

    fps_list = []
    last_print = time.time()

    print("Connecting to live stream... (Press 'q' in OpenCV window to quit)")

    try:
        while True:
            t_start = time.time()
            has_frame, frame, tele = capture.read()

            if not has_frame or frame is None:
                if not tele.get("connected"):
                    print(f"Waiting for stream connection... Reconnects: {tele.get('reconnect_count')}", end="\r")
                time.sleep(0.02)
                continue

            # Resize if excessive resolution to maintain 25+ FPS
            h, w = frame.shape[:2]
            if w > 1280:
                frame = cv2.resize(frame, (1280, int(h * 1280 / w)))

            # 3. Object Detection (Filtered to Vehicles)
            detections = detector.detect(frame)

            # 4. Metric Aggregation
            counts: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
            for d in detections:
                c = d.get("class_name", "car")
                if c in counts:
                    counts[c] += 1

            total_vehicles = len(detections)
            density_info = density_estimator.estimate(total_vehicles)
            density_label = density_info.get("level", "LOW")

            # FPS calculation
            t_end = time.time()
            instant_fps = 1.0 / max(1e-4, t_end - t_start)
            fps_list.append(instant_fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)

            # Terminal heartbeat every 2 seconds
            if time.time() - last_print > 2.0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] Proc: {avg_fps:.1f} FPS | "
                    f"Ingest: {tele.get('ingest_fps', 0):.1f} FPS | "
                    f"Vehicles: {total_vehicles} (🚗 {counts['car']}, 🏍️ {counts['motorcycle']}, 🚌 {counts['bus']}, 🚛 {counts['truck']}) | "
                    f"Density: {density_label}"
                )
                last_print = time.time()

            # 5. Render On-Screen HUD
            if not headless:
                hud_frame = draw_hud(
                    frame=frame,
                    detections=detections,
                    counts=counts,
                    density_label=density_label,
                    fps=avg_fps,
                    ingest_fps=tele.get("ingest_fps", 0.0),
                    source_name=str(stream_url)
                )

                cv2.imshow("RoadSentry AI - Live Traffic Camera Feed", hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:  # 'q' or ESC
                    print("\nUser requested exit.")
                    break

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        capture.stop()
        cv2.destroyAllWindows()
        print("Capture worker released and windows closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoadSentry AI - Live Traffic Camera Stream Monitor")
    parser.add_argument("--stream", type=str, default=STREAM_URL, help="HLS .m3u8, RTSP, HTTP, or video file path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="YOLOv8 weights path (e.g. yolov8n.pt)")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV GUI window (terminal stats only)")
    args = parser.parse_args()

    run_traffic_monitor(stream_url=args.stream, model_path=args.model, headless=args.headless)
