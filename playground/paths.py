"""Project and local-cache paths shared by every launcher."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path(
    os.environ.get("MODEL_PLAYGROUND_HOME", str(PROJECT_ROOT / ".local"))
).expanduser().resolve()
ASSETS_ROOT = LOCAL_ROOT / "assets"
HISTORY_ROOT = LOCAL_ROOT / "history"
WORK_ROOT = LOCAL_ROOT / "work"
MODELS_ROOT = ASSETS_ROOT / "models"
RUNTIMES_ROOT = ASSETS_ROOT / "runtimes"
DOWNLOADS_ROOT = ASSETS_ROOT / "downloads"
LOGS_ROOT = HISTORY_ROOT / "logs"
STATE_ROOT = WORK_ROOT / "state"
CACHE_ROOT = WORK_ROOT / "cache"


def ensure_local_layout() -> None:
    for directory in (MODELS_ROOT, RUNTIMES_ROOT, DOWNLOADS_ROOT, LOGS_ROOT,
                      HISTORY_ROOT / "images", HISTORY_ROOT / "input",
                      HISTORY_ROOT / "comfyui-user", STATE_ROOT, CACHE_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
