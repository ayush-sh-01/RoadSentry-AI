"""
RoadSentry AI - Sample Traffic Video Generator
Creates a synthetic road simulation MP4 with moving vehicles, lane markers, an ambulance with siren beacon, and an accident scenario.
Useful for immediate testing and demonstration without downloading large external clips.
"""

import os
import cv2
import numpy as np


def create_traffic_demo(
    output_path: str = "data/test_videos/demo_traffic.mp4",
    num_frames: int = 180,
    width: int = 854,
    height: int = 480,
    fps: int = 30
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Generating synthetic traffic video: {output_path} ({num_frames} frames)...")

    # Define vehicle agents: [x, y, vx, vy, color, w, h, is_ambulance, is_crash_target]
    vehicles = [
        # Normal moving cars lane 1 (downward)
        {"x": 280, "y": 30, "vx": 0, "vy": 3.5, "color": (200, 200, 200), "w": 40, "h": 65, "type": "car"},
        {"x": 350, "y": 140, "vx": 0, "vy": 4.0, "color": (50, 50, 220), "w": 42, "h": 70, "type": "car"},
        {"x": 290, "y": 280, "vx": 0, "vy": 3.2, "color": (220, 150, 50), "w": 45, "h": 75, "type": "car"},

        # Emergency vehicle (Ambulance with siren beacon)
        {"x": 420, "y": 10, "vx": 0, "vy": 5.5, "color": (245, 245, 245), "w": 48, "h": 85, "type": "ambulance"},

        # Crash pair lane 3 (they move then collide at frame 60)
        {"x": 510, "y": 50, "vx": 0, "vy": 3.8, "color": (80, 200, 80), "w": 42, "h": 68, "type": "crash_1"},
        {"x": 515, "y": 220, "vx": 0, "vy": 1.2, "color": (180, 50, 180), "w": 44, "h": 72, "type": "crash_2"},
    ]

    for frame_idx in range(num_frames):
        # Create road asphalt canvas
        frame = np.full((height, width, 3), 45, dtype=np.uint8)

        # Draw road borders and lanes
        road_left = 220
        road_right = 630
        cv2.rectangle(frame, (road_left, 0), (road_right, height), (60, 60, 60), -1)

        # White side curbs
        cv2.line(frame, (road_left, 0), (road_left, height), (255, 255, 255), 4)
        cv2.line(frame, (road_right, 0), (road_right, height), (255, 255, 255), 4)

        # Dashed yellow & white center dividers
        dash_offset = (frame_idx * 6) % 40
        for y in range(-dash_offset, height, 40):
            cv2.line(frame, (325, y), (325, y + 20), (200, 200, 200), 2)
            cv2.line(frame, (425, y), (425, y + 20), (0, 220, 255), 3)  # Yellow center line
            cv2.line(frame, (525, y), (525, y + 20), (200, 200, 200), 2)

        # Update and render vehicles
        for v in vehicles:
            # Crash dynamics simulation
            if v["type"] == "crash_1":
                if frame_idx < 65:
                    v["y"] += v["vy"]
                elif frame_idx < 75:
                    # Swerve right into car 2
                    v["x"] += 0.8
                    v["y"] += 1.5
                else:
                    # Stopped after collision
                    pass
            elif v["type"] == "crash_2":
                if frame_idx < 65:
                    v["y"] += v["vy"]
                else:
                    # Stopped after collision
                    pass
            else:
                v["y"] += v["vy"]
                if v["y"] > height:
                    v["y"] = -80

            vx, vy, vw, vh = int(v["x"]), int(v["y"]), v["w"], v["h"]

            # Vehicle chassis body
            cv2.rectangle(frame, (vx - vw // 2, vy - vh // 2), (vx + vw // 2, vy + vh // 2), v["color"], -1)
            cv2.rectangle(frame, (vx - vw // 2, vy - vh // 2), (vx + vw // 2, vy + vh // 2), (20, 20, 20), 2)

            # Windshield
            cv2.rectangle(frame, (vx - vw // 2 + 4, vy - vh // 4), (vx + vw // 2 - 4, vy - vh // 8), (20, 20, 30), -1)

            # If ambulance: draw flashing red/blue beacon on roof
            if v["type"] == "ambulance":
                siren_color1 = (0, 0, 255) if (frame_idx // 3) % 2 == 0 else (255, 0, 0)
                siren_color2 = (255, 0, 0) if (frame_idx // 3) % 2 == 0 else (0, 0, 255)
                # Red cross on chassis
                cv2.line(frame, (vx, vy + 5), (vx, vy + 25), (0, 0, 220), 4)
                cv2.line(frame, (vx - 10, vy + 15), (vx + 10, vy + 15), (0, 0, 220), 4)
                # Flashing beacons
                cv2.circle(frame, (vx - 8, vy - vh // 3), 5, siren_color1, -1)
                cv2.circle(frame, (vx + 8, vy - vh // 3), 5, siren_color2, -1)

        out.write(frame)

    out.release()
    print(f"Generated {output_path} successfully.")


if __name__ == "__main__":
    create_traffic_demo()
