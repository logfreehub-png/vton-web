"""Sozlamalar. Hammasi environment variable orqali o'zgartiriladi (run.sh ichida)."""
import os
from pathlib import Path

# Loyiha papkasi: vton-web/
BASE_DIR = Path(__file__).resolve().parent.parent

# ComfyUI manzili (Qwen-Image-Edit shu yerda ishlaydi)
COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188")
COMFY_DIR = Path(os.getenv("COMFY_DIR", str(Path.home() / "ComfyUI")))

# ~/ComfyUI/models/ ichidagi fayl nomlari
UNET = os.getenv("QWEN_UNET", "qwen-image-edit-2511-Q3_K_M.gguf")
TEXT_ENCODER = os.getenv("QWEN_TEXT_ENCODER", "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
VAE = os.getenv("QWEN_VAE", "qwen_image_vae.safetensors")
LORA = os.getenv("QWEN_LORA", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")

# Natija rasmining o'lchami (megapiksel). 1.0 ≈ 864x1152
MEGAPIXELS = float(os.getenv("QWEN_MEGAPIXELS", "1.0"))

# Bitta rasm uchun maksimal hajm (MB)
MAX_UPLOAD_MB = int(os.getenv("VTON_MAX_UPLOAD_MB", "15"))

# Natijalarni diskka ham saqlash (1 = ha, 0 = yo'q).
# Production'da 0: aks holda har bir mijoz surati outputs/ ichida muddatsiz qoladi.
SAVE_OUTPUTS = os.getenv("VTON_SAVE_OUTPUTS", "0") == "1"

# /api/tryon uchun maxfiy kalit (X-Api-Key sarlavhasi). Bo'sh bo'lsa tekshirilmaydi:
# faqat lokal sinov uchun. Serverda albatta bering, aks holda GPU hammaga ochiq.
API_KEY = os.getenv("VTON_API_KEY", "").strip()

STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "outputs"
