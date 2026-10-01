"""Paths, environment settings and feature flags."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]  # repo root
load_dotenv(ROOT / ".env")

DATA = ROOT / "data" / "robot"
EPISODES = DATA / "episodes"
SCANS = DATA / "scans"
VOICE_CACHE = DATA / "voice"
CERTS = ROOT / "certs"
STATIC = ROOT / "apps" / "robot_ui"
for p in (DATA, EPISODES, SCANS, VOICE_CACHE, CERTS):
    p.mkdir(parents=True, exist_ok=True)


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


PORT = int(os.getenv("HTTPS_PORT", "8443"))
HTTP_PORT = int(os.getenv("HTTP_PORT", "8000"))
ROBOT_NAME = os.getenv("ROBOT_NAME", "Pip")

AI_PROVIDER = os.getenv("AI_PROVIDER", "nvidia").lower()
AI_API_KEY = os.getenv("AI_API_KEY", "").strip()
AI_MODEL = os.getenv("AI_MODEL", "meta/muse-glimmer-30b")
AI_VISION_MODEL = os.getenv("AI_VISION_MODEL", "meta/muse-glimmer-30b")
# tried in order when the main model is down or fails (free endpoints come and go)
AI_FALLBACK_MODELS = [m.strip() for m in os.getenv("AI_FALLBACK_MODELS", "meta/llama-3.2-11b-vision-instruct").split(",") if m.strip()]

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")

MIN_DEMOS = int(os.getenv("MIN_DEMOS", "2"))
SEED_DEMOS = int(os.getenv("SEED_DEMOS", "4"))

FEATURE_CAMERA = _flag("FEATURE_CAMERA")
FEATURE_BRAIN = _flag("FEATURE_BRAIN")
FEATURE_SHAKE = _flag("FEATURE_SHAKE")
