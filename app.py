"""
RoadSentry AI - Interactive Web Dashboard
Autonomous Traffic Monitoring, Live Heatmap, Emergency Route Assistance, and Accident SOS Dispatch.
"""

from typing import Dict, Any, List, Optional
import os
import time
import tempfile
import inspect
import importlib
import yaml
import cv2
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# Force fresh module reload for all internal modules to prevent stale in-memory caching in long-running Streamlit processes
import sys
import importlib
import src.heatmap
import src.density
import src.tracker
import src.detector
import src.classifier_refiner
import src.accident_detector
import src.emergency_classifier
import src.route_advisor
import src.sos_notifier
import src.logger
import src.pipeline

for mod in [
    src.heatmap,
    src.density,
    src.tracker,
    src.detector,
    src.classifier_refiner,
    src.accident_detector,
    src.emergency_classifier,
    src.route_advisor,
    src.sos_notifier,
    src.logger,
    src.pipeline
]:
    try:
        importlib.reload(mod)
    except Exception:
        pass

from src.pipeline import RoadSentryPipeline

# Configure Streamlit page layout and theme
st.set_page_config(
    page_title="RoadSentry AI - Traffic, Route & Incident Management",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Persistent theme state in session_state (default to 'dark')
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"

current_theme = st.session_state["theme_mode"]

# Dynamic CSS for Dark / Light Theme Switching
if current_theme == "light":
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #1e293b;
        }

        .stApp {
            background: radial-gradient(circle at 15% 15%, rgba(2, 132, 199, 0.05) 0%, transparent 45%),
                        radial-gradient(circle at 85% 85%, rgba(99, 102, 241, 0.04) 0%, transparent 50%),
                        #f8fafc;
            color: #0f172a;
        }

        .app-header {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 18px 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.06);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .app-title {
            font-size: 26px;
            font-weight: 800;
            background: linear-gradient(135deg, #0f172a 0%, #0284c7 60%, #2563eb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
            margin: 0;
        }
        .app-subtitle {
            font-size: 13px;
            color: #64748b;
            margin-top: 4px;
            font-weight: 500;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 16px 14px;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 10px -1px rgba(15, 23, 42, 0.05);
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #0284c7, #38bdf8);
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #cbd5e1;
            box-shadow: 0 8px 20px -2px rgba(2, 132, 199, 0.12);
        }
        .metric-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 26px;
            font-weight: 700;
            color: #0284c7;
            margin-top: 6px;
            letter-spacing: -0.5px;
        }
        .metric-lbl {
            font-size: 11px;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        .metric-sub {
            font-size: 11px;
            color: #475569;
            margin-top: 5px;
            font-weight: 500;
        }

        .alert-card-critical {
            background: #fff1f2;
            border: 1px solid #fecdd3;
            border-left: 5px solid #e11d48;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(225, 29, 72, 0.08);
            color: #881337;
        }
        .alert-card-priority {
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-left: 5px solid #0284c7;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(2, 132, 199, 0.08);
            color: #0c4a6e;
        }
        .badge-critical {
            background: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
            color: white;
            padding: 3px 9px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 8px rgba(244, 63, 94, 0.25);
        }
        .badge-priority {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            color: #ffffff;
            padding: 3px 9px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 8px rgba(2, 132, 199, 0.25);
        }

        .sos-banner {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05);
            color: #1e293b;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #f1f5f9;
            padding: 6px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 18px;
            font-size: 13px;
            font-weight: 600;
            color: #64748b;
            border: none !important;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background: #ffffff !important;
            color: #0284c7 !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
        }

        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #0f172a;
            font-weight: 700;
            letter-spacing: -0.3px;
        }

        div.stButton > button {
            border-radius: 10px;
            font-weight: 600;
            font-size: 13px;
            letter-spacing: 0.3px;
            background: #ffffff;
            border: 1px solid #cbd5e1;
            color: #0f172a;
            transition: all 0.25s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            background: #f8fafc;
            border-color: #0284c7;
            color: #0284c7;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.15);
        }
    </style>
    """
else:
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #f8fafc;
        }

        .stApp {
            background: radial-gradient(circle at 15% 15%, rgba(0, 229, 255, 0.04) 0%, transparent 45%),
                        radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.04) 0%, transparent 50%),
                        #0b0f19;
            color: #f8fafc;
        }

        .app-header {
            background: linear-gradient(135deg, rgba(20, 26, 43, 0.75) 0%, rgba(15, 23, 42, 0.6) 100%);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 18px 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .app-title {
            font-size: 26px;
            font-weight: 800;
            background: linear-gradient(135deg, #f8fafc 0%, #00e5ff 60%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
            margin: 0;
        }
        .app-subtitle {
            font-size: 13px;
            color: #94a3b8;
            margin-top: 4px;
            font-weight: 500;
        }

        .metric-card {
            background: linear-gradient(145deg, rgba(26, 32, 53, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 16px 14px;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.6), transparent);
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(0, 229, 255, 0.3);
            box-shadow: 0 10px 28px -4px rgba(0, 229, 255, 0.15);
        }
        .metric-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 26px;
            font-weight: 700;
            color: #00e5ff;
            margin-top: 6px;
            letter-spacing: -0.5px;
            text-shadow: 0 0 20px rgba(0, 229, 255, 0.35);
        }
        .metric-lbl {
            font-size: 11px;
            font-weight: 700;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        .metric-sub {
            font-size: 11px;
            color: #cbd5e1;
            margin-top: 5px;
            font-weight: 500;
        }

        .alert-card-critical {
            background: linear-gradient(135deg, rgba(69, 10, 20, 0.65) 0%, rgba(20, 15, 25, 0.8) 100%);
            border: 1px solid rgba(244, 63, 94, 0.3);
            border-left: 5px solid #f43f5e;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 10px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 16px rgba(244, 63, 94, 0.15);
        }
        .alert-card-priority {
            background: linear-gradient(135deg, rgba(8, 47, 73, 0.65) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(0, 229, 255, 0.3);
            border-left: 5px solid #00e5ff;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 10px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 16px rgba(0, 229, 255, 0.15);
        }
        .badge-critical {
            background: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
            color: white;
            padding: 3px 9px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 8px rgba(244, 63, 94, 0.4);
        }
        .badge-priority {
            background: linear-gradient(135deg, #00e5ff 0%, #0284c7 100%);
            color: #041324;
            padding: 3px 9px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 8px rgba(0, 229, 255, 0.4);
        }

        .sos-banner {
            background: linear-gradient(145deg, rgba(22, 28, 48, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 16px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.3);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(15, 23, 42, 0.6);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 18px;
            font-size: 13px;
            font-weight: 600;
            color: #94a3b8;
            border: none !important;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(0, 229, 255, 0.15) 0%, rgba(14, 165, 233, 0.15) 100%) !important;
            color: #00e5ff !important;
            border: 1px solid rgba(0, 229, 255, 0.35) !important;
            box-shadow: 0 2px 10px rgba(0, 229, 255, 0.1);
        }

        section[data-testid="stSidebar"] {
            background-color: #0e1320;
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #f1f5f9;
            font-weight: 700;
            letter-spacing: -0.3px;
        }

        div.stButton > button {
            border-radius: 10px;
            font-weight: 600;
            font-size: 13px;
            letter-spacing: 0.3px;
            transition: all 0.25s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 18px rgba(0, 229, 255, 0.25);
        }
    </style>
    """

st.markdown(custom_css, unsafe_allow_html=True)


def load_settings() -> Dict[str, Any]:
    """Loads configuration settings from config/settings.yaml with safe fallbacks."""
    config_file = "config/settings.yaml"
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def create_traffic_map(
    lat: float,
    lon: float,
    location_name: str,
    density_level: str,
    vehicle_count: int,
    route_waypoints: Optional[List[Dict[str, Any]]] = None,
    is_simulated: bool = True
) -> pdk.Deck:
    """
    Renders an interactive dark-themed PyDeck map of the monitored road location
    with live color-coded traffic intensity and emergency route corridors.
    """
    color_map = {
        "LOW": [0, 230, 118, 220],     # Vibrant Green
        "MEDIUM": [255, 145, 0, 220],  # Amber/Orange
        "HIGH": [255, 23, 68, 220]     # Vibrant Red
    }
    rgba = color_map.get(density_level, [0, 230, 118, 220])
    radius = max(80, min(300, vehicle_count * 15 + 80))

    gps_label = "Simulated CCTV GPS Anchor" if is_simulated else "Live Hardware GPS"

    junction_data = [{
        "lat": lat,
        "lon": lon,
        "name": location_name,
        "density": density_level,
        "vehicles": vehicle_count,
        "status": gps_label
    }]

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=junction_data,
            get_position="[lon, lat]",
            get_color=rgba,
            get_radius=radius,
            pickable=True,
            filled=True,
            stroked=True,
            get_line_color=[255, 255, 255, 220],
            line_width_min_pixels=2
        )
    ]

    # Render Emergency Route Corridor if active
    if route_waypoints and len(route_waypoints) > 1:
        path_coords = [[w["lon"], w["lat"]] for w in route_waypoints]
        path_data = [{
            "path": path_coords,
            "name": "Emergency Priority Green Corridor"
        }]
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=path_data,
                get_path="path",
                get_color=[0, 229, 255, 240],  # Neon Cyan
                width_min_pixels=5,
                pickable=True
            )
        )

    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=14.5,
        pitch=25,
        bearing=0
    )

    is_light = st.session_state.get("theme_mode", "dark") == "light"
    map_style = "light" if is_light else "dark"
    tooltip_bg = "#ffffff" if is_light else "#1e2130"
    tooltip_color = "#0f172a" if is_light else "#f8fafc"
    tooltip_border = "1px solid #e2e8f0" if is_light else "none"

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=map_style,
        tooltip={
            "html": "<b>{name}</b><br/>Traffic Density: <b>{density}</b> ({vehicles} vehicles)<br/>GPS: {status}",
            "style": {"backgroundColor": tooltip_bg, "color": tooltip_color, "fontSize": "12px", "border": tooltip_border}
        }
    )


def main():
    settings = load_settings()
    loc_cfg = settings.get("location", {})
    sos_cfg = settings.get("sos", {})
    route_cfg = settings.get("route_assistance", {})
    accident_cfg = settings.get("accident_detector", {})
    density_cfg = settings.get("density", {})
    default_roi = density_cfg.get("default_roi", None)

    # Top Navigation Bar with Header & Theme Toggle Button
    top_col1, top_col2 = st.columns([6, 1])

    with top_col1:
        st.markdown("""
        <div class="app-header">
            <div>
                <div class="app-title">🚦 RoadSentry AI <span style="font-size:16px; font-weight:600; color:#0284c7; margin-left:8px; border:1px solid rgba(2,132,199,0.3); padding:3px 10px; border-radius:20px; background:rgba(2,132,199,0.08);">AUTONOMOUS SUITE</span></div>
                <div class="app-subtitle">Real-time Vision Analytics • Adaptive Congestion Heatmap • Emergency Priority Corridors • Automated Accident SOS</div>
            </div>
            <div style="text-align: right;">
                <span style="display:inline-flex; align-items:center; gap:6px; background:rgba(16,185,129,0.12); color:#10b981; padding:5px 12px; border-radius:8px; font-size:12px; font-weight:700; border:1px solid rgba(16,185,129,0.3);">
                    <span style="width:8px; height:8px; border-radius:50%; background:#10b981; box-shadow:0 0 10px #10b981;"></span>
                    SYSTEM ACTIVE
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with top_col2:
        is_dark = st.session_state["theme_mode"] == "dark"
        theme_btn_label = "☀️ Light" if is_dark else "🌙 Dark"
        if st.button(theme_btn_label, use_container_width=True, help="Toggle between Light and Dark visual appearance"):
            st.session_state["theme_mode"] = "light" if is_dark else "dark"
            st.rerun()

    # ---------------------------------------------------------
    # SIDEBAR CONTROLS
    # ---------------------------------------------------------
    st.sidebar.header("⚙️ System Controls")

    # Video Source
    input_type = st.sidebar.selectbox(
        "📹 Video Input Source",
        [
            "Demo Synthetic Traffic Clip",
            "Live Stream (HLS .m3u8 / RTSP / Snapshot)",
            "Upload Video File",
            "Webcam (Live)",
            "RTSP Stream"
        ],
        help="Select video stream. Use Live Stream for public HLS CCTV feeds, or Demo Clip to immediately test detection."
    )

    video_source = None
    use_threaded_stream = False
    if input_type == "Demo Synthetic Traffic Clip":
        demo_path = "data/test_videos/demo_traffic.mp4"
        if not os.path.exists(demo_path):
            st.warning("Creating demo traffic clip...")
            from scripts.generate_sample_video import create_traffic_demo
            create_traffic_demo(demo_path)
        video_source = demo_path
        st.sidebar.info("💡 **Demo Clip:** Features standard vehicles, a flashing ambulance, and a collision event for testing.")

    elif input_type == "Live Stream (HLS .m3u8 / RTSP / Snapshot)":
        from config.config import STREAM_URL
        custom_url = st.sidebar.text_input(
            "Live Stream URL (HLS / m3u8 / RTSP / Snapshot)",
            value=STREAM_URL,
            help="Paste public traffic camera HLS .m3u8 link, RTSP URL, or static JPEG snapshot URL."
        )
        video_source = custom_url.strip()
        use_threaded_stream = True
        st.sidebar.caption("⚡ Low-latency threaded capture worker active (zero buffer delay & auto-reconnect).")

    elif input_type == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI video", type=["mp4", "avi", "mov"], help="Upload any recorded CCTV traffic video.")
        if uploaded_file is not None:
            upload_dir = "data/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            cached_path = os.path.join(upload_dir, "current_input.mp4")

            # Store only once and reuse across reruns to prevent filling up disk space
            file_sig = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("uploaded_sig") != file_sig or not os.path.exists(cached_path):
                with open(cached_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state["uploaded_sig"] = file_sig

            video_source = cached_path

    elif input_type == "Webcam (Live)":
        cam_idx = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0)
        video_source = int(cam_idx)

    elif input_type == "RTSP Stream":
        rtsp_url = st.sidebar.text_input("Live CCTV Address (RTSP)", "rtsp://localhost:8554/live")
        video_source = rtsp_url
        use_threaded_stream = True

    st.sidebar.markdown("---")

    # Feature 4: SOS Settings Sidebar Section
    st.sidebar.subheader("🚨 Emergency SOS Mobile Settings")
    st.sidebar.caption("Where accident notifications will be sent:")

    contact_phone = st.sidebar.text_input(
        "📱 Emergency Phone Number",
        value=sos_cfg.get("recipient_phone", "+91-9244514953"),
        help="The phone number that receives the accident SMS and urgent push alerts."
    )

    with st.sidebar.expander("📲 Direct Telecom SMS Gateway (Fast2SMS / Twilio)", expanded=True):
        st.markdown("""
        <b>Direct SMS to Mobile:</b><br>
        For Indian numbers (+91), enter your free <a href="https://www.fast2sms.com" target="_blank">Fast2SMS</a> API key below. Fast2SMS gives free test credits upon signup.
        """, unsafe_allow_html=True)
        fast2sms_key = st.text_input(
            "Fast2SMS API Key (for India SMS)",
            value=sos_cfg.get("fast2sms_apikey", ""),
            type="password",
            placeholder="Paste your Fast2SMS dev key",
            help="Your API Authorization key from the Fast2SMS dashboard."
        )

    with st.sidebar.expander("🔔 Free Instant Phone Push Channel (Zero Setup)", expanded=True):
        st.markdown("""
        <b>Instant Lockscreen Notification:</b><br>
        Download the free <b>ntfy</b> app on your phone (<a href="https://play.google.com/store/apps/details?id=io.heckel.ntfy" target="_blank">Android</a> / <a href="https://apps.apple.com/app/ntfy/id1625396347" target="_blank">iOS</a>), and subscribe to your phone topic:
        """, unsafe_allow_html=True)
        cleaned_num = "".join(filter(str.isdigit, contact_phone)) or "alert"
        default_topic = sos_cfg.get("ntfy_topic", f"roadsentry_{cleaned_num}")
        ntfy_topic_input = st.text_input(
            "Phone Push Topic",
            value=default_topic,
            help="Topic name in the ntfy app. Alerts will pop up on your lock screen with siren sound."
        )

    with st.sidebar.expander("⚙️ Optional Fallback Channels (Email / Webhook)", expanded=False):
        contact_email = st.text_input(
            "Optional Fallback Email",
            value=sos_cfg.get("recipient_email", "ayuayu7697@gmail.com")
        )
        sender_email = st.text_input(
            "Sender Gmail Address",
            value=sos_cfg.get("smtp_user", ""),
            placeholder="e.g. sender.alerts@gmail.com"
        )
        sender_app_password = st.text_input(
            "Google 16-character App Password",
            value=sos_cfg.get("smtp_password", ""),
            type="password"
        )
        custom_webhook = st.text_input(
            "Optional Discord / Telegram Webhook",
            value=sos_cfg.get("webhook_url", "")
        )

    test_sos_btn = st.sidebar.button("🧪 Send Test SOS to My Number", help="Dispatches a test notification to verify delivery to your phone.")

    st.sidebar.markdown("""
    <div style="font-size: 11px; color: #64748b; background-color: #f1f5f9; border: 1px solid #e2e8f0; padding: 8px; border-radius: 4px; margin-top: 4px;">
        📍 <b>Permanent Config:</b> You can also save credentials permanently in <code>config/settings.yaml</code> under the <code>sos:</code> section.
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # Feature 2 & 3: Monitored Location Settings
    st.sidebar.subheader("📍 Monitored Road & GPS Settings")
    road_name = st.sidebar.text_input("Junction Name", value=loc_cfg.get("junction_name", "Central Ring Road - Hospital Junction"))
    loc_lat = st.sidebar.number_input("Latitude", value=float(loc_cfg.get("latitude", 12.9716)), format="%.4f")
    loc_lon = st.sidebar.number_input("Longitude", value=float(loc_cfg.get("longitude", 77.5946)), format="%.4f")
    is_sim_gps = st.sidebar.checkbox("Simulated CCTV GPS Location", value=bool(loc_cfg.get("is_simulated_gps", True)), help="Clearly labels that coordinates are a fixed CCTV anchor if the camera feed lacks an embedded GPS stream.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("👁️ Visual Overlays")
    show_tracks = st.sidebar.checkbox("Vehicle Bounding Boxes & IDs", value=True)
    show_roi = st.sidebar.checkbox("Traffic Monitoring Zone (ROI)", value=True)
    show_emergency = st.sidebar.checkbox("Emergency Vehicle Priority Badges", value=True)
    show_accidents = st.sidebar.checkbox("Accident Collision Warnings", value=True)

    st.sidebar.markdown("---")
    if "is_monitoring" not in st.session_state:
        st.session_state["is_monitoring"] = False

    start_stream = st.sidebar.button("▶️ Start Monitoring", type="primary", use_container_width=True)
    stop_stream = st.sidebar.button("⏹️ Stop Monitoring", use_container_width=True)

    if start_stream:
        st.session_state["is_monitoring"] = True
    if stop_stream:
        st.session_state["is_monitoring"] = False

    # ---------------------------------------------------------
    # TOP KPI METRICS BAR (Features 1, 2, 3, 4)
    # ---------------------------------------------------------
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    fps_metric = kpi_col1.empty()
    density_metric = kpi_col2.empty()
    counts_metric = kpi_col3.empty()
    emergency_metric = kpi_col4.empty()
    sos_metric = kpi_col5.empty()

    # Initial KPI cards
    fps_metric.markdown('<div class="metric-card"><div class="metric-lbl">Processing Speed</div><div class="metric-val">0.0 FPS</div><div class="metric-sub">Vision Pipeline</div></div>', unsafe_allow_html=True)
    density_metric.markdown('<div class="metric-card"><div class="metric-lbl">Traffic Density</div><div class="metric-val" style="color:#00e676">LOW</div><div class="metric-sub">Smooth Free Flow</div></div>', unsafe_allow_html=True)
    counts_metric.markdown('<div class="metric-card"><div class="metric-lbl">Vehicles in View</div><div class="metric-val">0</div><div class="metric-sub">Cars: 0 | Bikes: 0 | Trucks: 0</div></div>', unsafe_allow_html=True)
    emergency_metric.markdown('<div class="metric-card"><div class="metric-lbl">Emergency Vehicles</div><div class="metric-val" style="color:#00e5ff">0</div><div class="metric-sub">Corridors Standby</div></div>', unsafe_allow_html=True)
    sos_metric.markdown('<div class="metric-card"><div class="metric-lbl">Accidents & SOS</div><div class="metric-val" style="color:#00e676">0 Events</div><div class="metric-sub">SOS Ready</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # MAIN DASHBOARD PANELS
    # ---------------------------------------------------------
    left_col, right_col = st.columns([1.5, 1.1])

    with left_col:
        video_tabs = st.tabs(["📹 Live CCTV Camera Feed", "🔥 Road Congestion Heatmap"])
        with video_tabs[0]:
            video_frame_placeholder = st.empty()
            st.caption("Live CCTV video feed with active vehicle tracking, classifications, and instant hazard alerts.")
        with video_tabs[1]:
            heatmap_status_placeholder = st.empty()
            heatmap_placeholder = st.empty()
            st.markdown("""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin-top: 8px; font-size: 12px; color: #334155;">
                <b>🗺️ Heatmap Guide:</b> 🔴 <b>Red/Amber</b> = Stopped or crawling traffic (severe bottleneck). 🟢 <b>Green/Cyan</b> = Smoothly moving vehicles. Older trails fade automatically.
            </div>
            """, unsafe_allow_html=True)

    with right_col:
        core_tabs = st.tabs([
            "🗺️ Live Traffic Heatmap",
            "🚨 Emergency & Route Assist",
            "💥 Accident Detection & SOS"
        ])

        with core_tabs[0]:
            # Feature 2 & 1: Live Traffic Map
            st.subheader("Monitored Road Network Heatmap")
            map_gps_badge = '<span style="color:#ffb74d; font-size:12px; font-weight:600;">[⚠️ Simulated CCTV GPS Anchor]</span>' if is_sim_gps else '<span style="color:#00e676; font-size:12px;">[📡 Live Hardware GPS]</span>'
            map_label_placeholder = st.empty()
            map_label_placeholder.markdown(f"<b>📍 Monitored Road:</b> {road_name} {map_gps_badge}", unsafe_allow_html=True)
            map_placeholder = st.empty()

            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            st.subheader("Vehicle Category Breakdown")
            category_breakdown_placeholder = st.empty()

        with core_tabs[1]:
            # Feature 3: Emergency Vehicle Detection & Route Assistance
            st.subheader("Route & Intersection Assistance")
            st.caption("Automated traffic signal preemption and corridor clearance for emergency vehicles:")
            emergency_banner_placeholder = st.empty()

        with core_tabs[2]:
            # Feature 4: Accident Detection & SOS Alerts
            st.subheader("Accident Alerts & SOS Dispatch")
            sos_status_banner_placeholder = st.empty()
            st.caption("Immediate collision detection and automated SOS dispatch to your emergency contact:")
            accident_alerts_placeholder = st.empty()

    st.markdown("---")
    st.subheader("📋 Recorded Incident & Dispatch History")
    st.caption("Full audit log of detected collisions, emergency vehicle passes, and SOS dispatch timestamps:")
    table_placeholder = st.empty()
    exp_col1, exp_col2 = st.columns(2)

    # ---------------------------------------------------------
    # PIPELINE INITIALIZATION & EXECUTION
    # ---------------------------------------------------------
    location_payload = {
        "camera_id": loc_cfg.get("camera_id", "CAM-01"),
        "junction_name": road_name,
        "latitude": loc_lat,
        "longitude": loc_lon,
        "is_simulated_gps": is_sim_gps
    }

    eff_smtp_user = sender_email.strip() if sender_email.strip() else sos_cfg.get("smtp_user", "").strip()
    eff_smtp_pass = sender_app_password.strip().replace(" ", "") if sender_app_password.strip() else sos_cfg.get("smtp_password", "").strip().replace(" ", "")

    eff_fast2sms_key = fast2sms_key.strip() if 'fast2sms_key' in locals() and fast2sms_key.strip() else sos_cfg.get("fast2sms_apikey", "").strip()

    sos_payload = {
        "recipient_email": contact_email.strip() if 'contact_email' in locals() else sos_cfg.get("recipient_email", ""),
        "recipient_phone": contact_phone.strip(),
        "smtp_server": sos_cfg.get("smtp_server", "smtp.gmail.com"),
        "smtp_port": int(sos_cfg.get("smtp_port", 587)),
        "smtp_user": eff_smtp_user,
        "smtp_password": eff_smtp_pass,
        "cooldown_seconds": int(sos_cfg.get("cooldown_seconds", 45)),
        "enabled": bool(sos_cfg.get("enabled", True)),
        "fast2sms_apikey": eff_fast2sms_key,
        "twilio_account_sid": sos_cfg.get("twilio_account_sid", ""),
        "twilio_auth_token": sos_cfg.get("twilio_auth_token", ""),
        "twilio_from_number": sos_cfg.get("twilio_from_number", ""),
        "ntfy_topic": ntfy_topic_input.strip() if 'ntfy_topic_input' in locals() else sos_cfg.get("ntfy_topic", ""),
        "webhook_url": custom_webhook.strip() if 'custom_webhook' in locals() else sos_cfg.get("webhook_url", "")
    }

    # Handle manual Test SOS button click
    if test_sos_btn:
        from src.sos_notifier import SOSNotifier
        test_notifier = SOSNotifier(**sos_payload)
        t_res = test_notifier.test_dispatch(contact_email if 'contact_email' in locals() else None)
        if t_res.get("delivered"):
            st.sidebar.success(f"{t_res['message']}")
        elif t_res.get("status") == "SIMULATED_ONLY":
            st.sidebar.warning(f"{t_res['message']}")
        else:
            st.sidebar.error(f"{t_res['message']}")

    # Render static initial PyDeck map
    initial_map = create_traffic_map(
        lat=loc_lat,
        lon=loc_lon,
        location_name=road_name,
        density_level="LOW",
        vehicle_count=0,
        is_simulated=is_sim_gps
    )
    map_placeholder.pydeck_chart(initial_map, use_container_width=True)

    # Initial Route Assist State
    emergency_banner_placeholder.markdown("""
    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <b style="color: #1e293b;">SIGNAL STATUS: NORMAL CYCLIC ROTATION</b>
            <span style="background-color: #e2e8f0; color: #475569; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">STANDBY</span>
        </div>
        <div style="font-size: 12px; color: #64748b; margin-top: 6px;">
            Standard 45s cyclic timing. Priority green corridor on standby for ambulances or police.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initial SOS Delivery Banner
    sos_status_banner_placeholder.markdown(f"""
    <div class="sos-banner">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <b style="color: #0f172a;">📡 SOS DISPATCHER: ACTIVE</b>
            <span style="background-color: #10b981; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">READY</span>
        </div>
        <div style="font-size: 12px; color: #64748b; margin-top: 6px;">
            Target Emergency Mobile: <b style="color: #0284c7; font-size: 13px;">{contact_phone}</b><br>
            <i>Accidents immediately dispatch automated SOS SMS with live GPS anchor and vehicle telemetry.</i>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("is_monitoring", False) and video_source is not None:
        threaded_cap = None
        cap = None

        if use_threaded_stream:
            from src.stream_capture import ThreadedStreamCapture
            threaded_cap = ThreadedStreamCapture(source=video_source).start()
        else:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                st.error(f"Could not open video source: {video_source}. Please verify file path or camera connection.")
                return

        # Prepare pipeline kwargs safely with parameter inspection
        pipeline_kwargs = {
            "conf_thresh": 0.20,
            "iou_thresh": 0.65,
            "low_density_thresh": 2,
            "medium_density_thresh": 4,
            "accident_iou_thresh": 0.25,
            "heatmap_decay": 0.96,
            "enable_emergency_detection": show_emergency,
            "roi_polygon": default_roi,
            "accident_config": accident_cfg,
            "device": "cpu"
        }
        init_params = inspect.signature(RoadSentryPipeline.__init__).parameters
        if "location_data" in init_params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params.values()):
            pipeline_kwargs["location_data"] = location_payload
        if "sos_config" in init_params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params.values()):
            pipeline_kwargs["sos_config"] = sos_payload
        if "route_config" in init_params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params.values()):
            pipeline_kwargs["route_config"] = route_cfg

        pipeline = RoadSentryPipeline(**pipeline_kwargs)

        seen_accident_ids = set()
        seen_emergency_ids = set()
        frame_count = 0
        stream_running = True

        while stream_running:
            if threaded_cap is not None:
                has_frame, frame, stream_tele = threaded_cap.read()
                if not has_frame or frame is None:
                    time.sleep(0.02)
                    continue
            else:
                if not cap.isOpened():
                    break
                ret, frame = cap.read()
                if not ret:
                    if input_type == "Demo Synthetic Traffic Clip":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

            frame_count += 1
            h, w = frame.shape[:2]
            if w > 1280:
                frame = cv2.resize(frame, (1280, int(h * 1280 / w)))

            # Run Vision Pipeline
            annotated_frame, telemetry = pipeline.process_frame(
                frame,
                show_tracks=show_tracks,
                show_trails=False,
                show_roi=show_roi,
                show_heatmap=False,
                show_accidents=show_accidents,
                show_emergency=show_emergency
            )

            # Deduplicate unique events
            for acc in telemetry.get("accidents", []):
                seen_accident_ids.add(acc.get("incident_id"))
            for em in telemetry.get("emergencies", []):
                seen_emergency_ids.add(em.get("track_id"))

            total_accidents = len(seen_accident_ids)
            total_emergencies = len(seen_emergency_ids)

            # -----------------------------------------------------
            # UPDATE TOP KPI CARDS
            # -----------------------------------------------------
            d_state = telemetry["density_level"]
            d_color = "#00e676" if d_state == "LOW" else ("#ff9100" if d_state == "MEDIUM" else "#ff1744")
            d_sub = "Smooth Flow" if d_state == "LOW" else ("Moderate Traffic" if d_state == "MEDIUM" else "Heavy Congestion")

            counts_by_cls = telemetry.get("counts_by_class", {})
            car_cnt = counts_by_cls.get("car", 0)
            bike_cnt = counts_by_cls.get("motorcycle", 0)
            truck_cnt = counts_by_cls.get("truck", 0)
            bus_cnt = counts_by_cls.get("bus", 0)

            active_emer = len(telemetry.get("emergencies", []))
            emer_color = "#ff1744" if active_emer > 0 else "#00e5ff"
            emer_sub = "🚨 CORRIDOR ACTIVE" if active_emer > 0 else "Priority Standby"

            active_acc = len(telemetry.get("accidents", []))
            acc_color = "#ff1744" if total_accidents > 0 else "#00e676"
            sos_info = telemetry.get("sos_status", {})
            sos_deliv_txt = "✅ Delivered" if sos_info.get("delivered") else "Ready"

            fps_metric.markdown(f'<div class="metric-card"><div class="metric-lbl">Processing Speed</div><div class="metric-val">{telemetry["fps"]:.1f} FPS</div><div class="metric-sub">Real-Time Engine</div></div>', unsafe_allow_html=True)
            density_metric.markdown(f'<div class="metric-card"><div class="metric-lbl">Traffic Density</div><div class="metric-val" style="color:{d_color}">{d_state}</div><div class="metric-sub">{d_sub}</div></div>', unsafe_allow_html=True)
            counts_metric.markdown(f'<div class="metric-card"><div class="metric-lbl">Vehicles in View</div><div class="metric-val">{telemetry["vehicle_count"]}</div><div class="metric-sub">🚗 {car_cnt} | 🏍️ {bike_cnt} | 🚛 {truck_cnt} | 🚌 {bus_cnt}</div></div>', unsafe_allow_html=True)
            emergency_metric.markdown(f'<div class="metric-card"><div class="metric-lbl">Emergency Vehicles</div><div class="metric-val" style="color:{emer_color}">{total_emergencies}</div><div class="metric-sub">{emer_sub}</div></div>', unsafe_allow_html=True)
            sos_metric.markdown(f'<div class="metric-card"><div class="metric-lbl">Accidents & SOS</div><div class="metric-val" style="color:{acc_color}">{total_accidents} Events</div><div class="metric-sub">SOS: {sos_deliv_txt}</div></div>', unsafe_allow_html=True)

            # -----------------------------------------------------
            # RENDER VIDEO & CCTV HEATMAP
            # -----------------------------------------------------
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Hotspot Heatmap
            stats = telemetry.get("heatmap_stats", {"bottleneck_level": "CLEAR", "bottleneck_desc": "Flowing smoothly", "peak_intensity": 0})
            status_color = "#10b981" if stats["bottleneck_level"] == "CLEAR" else ("#f59e0b" if stats["bottleneck_level"] == "MODERATE" else "#ef4444")
            heatmap_status_placeholder.markdown(
                f'<div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid {status_color}; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">'
                f'<span style="font-weight: 700; color: {status_color};">HOTSPOT STATUS: {stats["bottleneck_level"]}</span> &nbsp;|&nbsp; '
                f'<span style="color: #1e293b; font-size: 13px;">{stats["bottleneck_desc"]}</span> '
                f'<span style="color: #64748b; font-size: 12px;">(Congestion Intensity: {stats["peak_intensity"]}%)</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            heatmap_preview = telemetry.get("heatmap_preview", telemetry.get("heatmap_raw"))
            heatmap_rgb = cv2.cvtColor(heatmap_preview, cv2.COLOR_BGR2RGB)
            heatmap_placeholder.image(heatmap_rgb, channels="RGB", use_container_width=True)

            # -----------------------------------------------------
            # FEATURE 2: UPDATE GEOGRAPHIC TRAFFIC MAP (Every 3 frames)
            # -----------------------------------------------------
            if frame_count % 3 == 0:
                route_data = telemetry.get("route_assistance", {})
                wps = route_data.get("corridor_waypoints", []) if route_data.get("is_active") else []

                live_map = create_traffic_map(
                    lat=loc_lat,
                    lon=loc_lon,
                    location_name=road_name,
                    density_level=d_state,
                    vehicle_count=telemetry["vehicle_count"],
                    route_waypoints=wps,
                    is_simulated=is_sim_gps
                )
                map_placeholder.pydeck_chart(live_map, use_container_width=True)

                # Feature 1: Category breakdown display
                total_v = max(1, telemetry["vehicle_count"])
                car_pct = int((car_cnt / total_v) * 100)
                bike_pct = int((bike_cnt / total_v) * 100)
                truck_pct = int((truck_cnt / total_v) * 100)
                bus_pct = int((bus_cnt / total_v) * 100)

                category_breakdown_placeholder.markdown(f"""
                <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); color: #1e293b;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 13px;">
                        <span>🚗 <b>Passenger Cars:</b> {car_cnt}</span>
                        <span style="color: #0284c7; font-weight: 600;">{car_pct}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 13px;">
                        <span>🏍️ <b>Motorcycles & Two-Wheelers:</b> {bike_cnt}</span>
                        <span style="color: #0284c7; font-weight: 600;">{bike_pct}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 13px;">
                        <span>🚛 <b>Commercial Trucks:</b> {truck_cnt}</span>
                        <span style="color: #0284c7; font-weight: 600;">{truck_pct}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 13px;">
                        <span>🚌 <b>Public Transit Buses:</b> {bus_cnt}</span>
                        <span style="color: #0284c7; font-weight: 600;">{bus_pct}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # -----------------------------------------------------
            # FEATURE 3: EMERGENCY & ROUTE ASSISTANCE PANEL
            # -----------------------------------------------------
            route_data = telemetry.get("route_assistance", {})
            if route_data.get("is_active"):
                emergency_banner_placeholder.markdown(f"""
                <div style="background-color: #f0f9ff; border: 2px solid #0284c7; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(2,132,199,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="badge-priority">PRIORITY CORRIDOR ACTIVE</span>
                        <b style="color: #0284c7; font-size: 16px;">{route_data.get('vehicle_type')} #{route_data.get('vehicle_id')}</b>
                    </div>
                    <div style="margin-top: 10px; font-size: 14px; font-weight: 700; color: #0f172a;">
                        🚦 {route_data.get('intersection_action')}
                    </div>
                    <div style="margin-top: 8px; font-size: 13px; color: #334155;">
                        🛣️ <b>Priority Corridor:</b> {route_data.get('corridor_name')}<br>
                        ⏱️ <b>Green Phase Extension:</b> {route_data.get('green_duration')}<br>
                        🛑 <b>Intersection Control:</b> {route_data.get('cross_traffic_status')}<br>
                        🏥 <b>Clear Destination:</b> {route_data.get('target_destination')}
                    </div>
                    <div style="margin-top: 10px; padding: 8px 12px; background: #e0f2fe; border-left: 3px solid #0284c7; border-radius: 4px; font-size: 12px; color: #0369a1;">
                        ⚠️ <b>Civilian Detour Advisory:</b> {route_data.get('detour_advisory')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                emergency_banner_placeholder.markdown("""
                <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <b style="color: #1e293b;">SIGNAL STATUS: NORMAL CYCLIC ROTATION</b>
                        <span style="background-color: #e2e8f0; color: #475569; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">STANDBY</span>
                    </div>
                    <div style="font-size: 12px; color: #64748b; margin-top: 6px;">
                        Standard 45s cyclic rotation active. Priority corridor on standby for ambulances, fire engines, or police vehicles.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # -----------------------------------------------------
            # FEATURE 4: ACCIDENT DETECTION & SOS STATUS PANEL
            # -----------------------------------------------------
            sos_status = telemetry.get("sos_status", {})
            sos_color = "#10b981" if sos_status.get("delivered") else "#f59e0b"
            sos_badge = "DELIVERED" if sos_status.get("delivered") else "STANDBY"

            sos_status_banner_placeholder.markdown(f"""
            <div class="sos-banner">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <b style="color: #0f172a;">📡 SOS DISPATCH STATUS: <span style="color: {sos_color};">{sos_badge}</span></b>
                    <span style="background-color: {sos_color}; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">{sos_status.get('status', 'STANDBY')}</span>
                </div>
                <div style="font-size: 12px; color: #475569; margin-top: 6px;">
                    {sos_status.get('message', 'Monitoring road for collision anomalies.')}
                </div>
                <div style="font-size: 11px; color: #64748b; margin-top: 6px; border-top: 1px solid #e2e8f0; padding-top: 4px;">
                    Target Emergency Mobile: <b style="color: #0284c7; font-size: 12px;">{contact_phone}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Render recent incidents feed
            recent_incidents = pipeline.logger.get_recent_incidents(limit=4)
            if recent_incidents:
                alerts_html = ""
                for inc in recent_incidents:
                    is_crit = inc.get("severity") == "CRITICAL"
                    card_cls = "alert-card-critical" if is_crit else "alert-card-priority"
                    badge_cls = "badge-critical" if is_crit else "badge-priority"
                    alerts_html += f"""
                    <div class="{card_cls}">
                        <div style="display: flex; justify-content: space-between;">
                            <span class="{badge_cls}">{inc.get('severity')}</span>
                            <span style="color: #94a3b8; font-size: 12px;">{inc.get('timestamp')}</span>
                        </div>
                        <div style="margin-top: 4px; font-weight: 700; color: #f8fafc; font-size: 14px;">
                            {inc.get('type')}
                        </div>
                        <div style="font-size: 12px; color: #cbd5e1; margin-top: 2px;">
                            Vehicles: #{inc.get('vehicle_ids')} &nbsp;|&nbsp; {inc.get('details')}
                        </div>
                    </div>
                    """
                accident_alerts_placeholder.markdown(alerts_html, unsafe_allow_html=True)
            else:
                accident_alerts_placeholder.info("Normal traffic conditions. No accidents detected.")

            # Incident History Table
            df = pipeline.logger.get_dataframe()
            table_placeholder.dataframe(df.head(15), use_container_width=True)

        if threaded_cap is not None:
            threaded_cap.stop()
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    else:
        # Show static snapshot of incident log if not streaming
        csv_file = "data/incident_logs.csv"
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                table_placeholder.dataframe(df.head(20), use_container_width=True)
                exp_col1.download_button(
                    "📥 Download Incident Log (CSV)",
                    df.to_csv(index=False),
                    file_name="roadsentry_incidents.csv",
                    mime="text/csv"
                )
                exp_col2.download_button(
                    "📥 Download Incident Log (JSON)",
                    df.to_json(orient="records", indent=2),
                    file_name="roadsentry_incidents.json",
                    mime="application/json"
                )
            except Exception:
                pass


if __name__ == "__main__":
    main()
