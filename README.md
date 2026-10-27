# RoadSentry AI: AI-Enabled Real-Time Road Scene Analytics and Incident Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

RoadSentry AI is a real-time computer vision system that analyzes live or recorded road-camera CCTV footage to detect vehicles, estimate traffic density, identify accidents and emergency vehicles, maintain cumulative traffic heat-maps, and surface alerts on an interactive web dashboard.

---

## 🌟 Four Core Features

1. **Traffic Monitoring:**
   - Detects and classifies vehicles (`car`, `motorcycle`, `bus`, `truck`) with temporal track smoothing.
   - Real-time total and category-wise vehicle counts.
   - Dynamic traffic density estimation (`LOW`, `MEDIUM`, `HIGH`) with road area occupancy.

2. **Live Traffic Heatmap:**
   - Color-coded interactive map (`carto-darkmatter` / OpenStreetMap) showing live traffic intensity at monitored road locations.
   - Clearly labels simulated locations if GPS hardware is unavailable (e.g. `[⚠️ Simulated CCTV GPS Anchor]`).
   - Thermal CCTV road pavement heatmap highlighting stopped vehicle queues and bottlenecks.

3. **Emergency Vehicle Detection & Route Assistance:**
   - Detects Ambulances, Fire Engines, and Police Cruisers via dual-strobe beacon analysis.
   - Automatic Route & Intersection Assistance: Green wave signal preemption, corridor clearance, and civilian detour advisories.

4. **Accident Detection & SOS Alerts:**
   - Detects potential road collisions via IoU overlap spikes and rapid deceleration.
   - Dispatches automated SOS notifications to your configured email and phone number with live delivery status.
   - Maintains full incident history with CSV/JSON exports.

---

## 🚨 Where to Configure Your SOS Email and Phone Number

You can configure your emergency email and phone number in two convenient places:

1. **Directly in the Web Dashboard Sidebar:**
   - In the Streamlit app sidebar under **"🚨 Emergency SOS Contact Settings"**, type your email in **"SOS Recipient Email"** and phone in **"Emergency Phone / SMS Number"**.
   - Click **"🧪 Send Test SOS Alert"** to verify delivery.

2. **Permanently in `config/settings.yaml`:**
   - Open [`config/settings.yaml`](file:///c:/Users/LENOVO/Desktop/RoadSentry%20AI/config/settings.yaml) and edit the `sos:` section:
   ```yaml
   sos:
     enabled: true
     recipient_email: "your_email@example.com"    # Change to your email address
     recipient_phone: "+91-9876543210"           # Change to your mobile number
     smtp_server: "smtp.gmail.com"
     smtp_port: 587
     smtp_user: ""                               # Your sender email (optional for live SMTP)
     smtp_password: ""                           # Your app password (optional for live SMTP)
   ```

---

## 📁 Repository Structure

```
RoadSentry AI/
├── config/
│   └── settings.yaml              # SOS credentials, GPS location, and detection thresholds
├── data/
│   ├── test_videos/               # Sample CCTV clips & synthetic traffic demos
│   ├── snapshots/                 # Incident snapshot crops
│   └── incident_logs.csv          # Persisted incident history
├── scripts/
│   └── generate_sample_video.py   # Benchmark synthetic traffic generator
├── src/
│   ├── __init__.py
│   ├── detector.py                # YOLOv8 vehicle detection wrapper
│   ├── tracker.py                 # ByteTrack multi-object tracker & speeds
│   ├── classifier_refiner.py      # Vehicle classification disambiguation
│   ├── density.py                 # Traffic density & road space occupancy
│   ├── heatmap.py                 # 2D decaying spatial heatmap generator
│   ├── accident_detector.py       # Collision & deceleration heuristics
│   ├── emergency_classifier.py    # Siren beacon & emergency vehicle detection
│   ├── sos_notifier.py            # Automated SOS email/SMS dispatcher
│   ├── route_advisor.py           # Signal preemption & emergency route assistance
│   ├── logger.py                  # CSV/JSON incident logging & snapshots
│   └── pipeline.py                # Master vision analytics pipeline orchestrator
├── tests/
│   ├── test_pipeline.py           # Unit tests for pipeline, SOS, route assist & classifier
│   └── test_classifier_refiner.py # Unit tests for vehicle classification
├── app.py                         # Simplified Streamlit web dashboard
├── requirements.txt               # Dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment
Ensure your Python 3.11 virtual environment is active:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Run Component Unit Tests
```powershell
python -m unittest tests/test_pipeline.py
```

### 3. Generate Sample Benchmark Video (Optional)
```powershell
python scripts/generate_sample_video.py
```

### 4. Launch the Interactive Web Dashboard
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
- Choose **Demo Synthetic Traffic Clip** or upload an MP4/CCTV video.
- Toggle overlays (Bounding boxes, Trajectories, ROI, Heatmap, Accidents).
- Click **▶️ Start Monitoring**.
