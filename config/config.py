"""
RoadSentry AI - Live Stream & Camera Configuration Loader
Provides unified access to stream URLs, capture settings, and detection configurations.
"""

import os
from typing import Dict, Any, List
import yaml

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(CONFIG_DIR, "settings.yaml")


def load_yaml_config(filepath: str = SETTINGS_FILE) -> Dict[str, Any]:
    """Loads configuration from YAML file with fallback defaults."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    return cfg
        except Exception:
            pass
    return {}


# Load raw settings
_settings = load_yaml_config()
_live_cfg = _settings.get("live_stream", {})
_model_cfg = _settings.get("model", {})

# Stream configuration (supports environment variable overrides)
STREAM_URL: str = os.environ.get(
    "ROADSENTRY_STREAM_URL",
    _live_cfg.get("stream_url", "https://cctv.austintexas.gov/m3u8/330.m3u8")
)
STREAM_TYPE: str = _live_cfg.get("stream_type", "auto")
RECONNECT_TIMEOUT_SEC: float = float(_live_cfg.get("reconnect_timeout_seconds", 5.0))
MAX_RECONNECT_ATTEMPTS: int = int(_live_cfg.get("max_reconnect_attempts", 10))
CAPTURE_READ_FPS: int = int(_live_cfg.get("read_fps", 25))
BUFFER_SIZE: int = int(_live_cfg.get("buffer_size", 1))

# Target traffic classes: 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASSES: List[int] = _model_cfg.get("target_classes", [1, 2, 3, 5, 7])
DEFAULT_MODEL: str = _model_cfg.get("detection_model", "yolov8n.pt")
CONF_THRESH: float = float(_model_cfg.get("confidence_threshold", 0.25))
IOU_THRESH: float = float(_model_cfg.get("iou_threshold", 0.65))
DEVICE: str = _model_cfg.get("device", "cpu")
