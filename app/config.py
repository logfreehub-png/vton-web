"""Sozlamalar. Hammasi environment variable orqali o'zgartiriladi (run.sh ichida)."""
import os
from pathlib import Path

# Loyiha papkasi: vton-web/
BASE_DIR = Path(__file__).resolve().parent.parent

# Model og'irliklari turgan papka (model.safetensors va dwpose/)
WEIGHTS_DIR = os.getenv(
    "VTON_WEIGHTS_DIR",
    str(Path.home() / "fashn-vton-1.5" / "weights"),
)

# "cuda" yoki "cpu". Bo'sh qolsa, model o'zi GPU'ni topadi.
DEVICE = os.getenv("VTON_DEVICE") or None

# Bitta rasm uchun maksimal hajm (MB)
MAX_UPLOAD_MB = int(os.getenv("VTON_MAX_UPLOAD_MB", "15"))

# Natijalarni diskka ham saqlash (1 = ha, 0 = yo'q)
SAVE_OUTPUTS = os.getenv("VTON_SAVE_OUTPUTS", "1") == "1"

STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "outputs"
