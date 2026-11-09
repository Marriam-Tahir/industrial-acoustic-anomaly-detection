"""Application settings. Training parameters live exclusively in the model export."""
from dataclasses import dataclass, field
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    export_dir: Path = field(default_factory=lambda: Path(os.environ.get(
        "SMART_LISTENER_EXPORT_DIR", str(ROOT / "models" / "convae_v1" / "export"))))
    max_upload_bytes: int = 25 * 1024 * 1024
    max_duration_seconds: float = 60.0
    min_duration_seconds: float = 1.0
    host: str = "127.0.0.1"
    port: int = 5000


MACHINE_TYPES = ("fan", "pump")
NOISE_LEVELS = ("6_dB", "0_dB", "-6_dB")
VERSION = "1.0.0"
