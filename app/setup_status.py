"""Model fayllari yuklanish holati (interfeysda real vaqtda ko'rsatish uchun).

Har bir fayl uchun diskdagi hajmni o'qiydi: tayyor fayl, yuklanayotgan
bo'laklar (<fayl>.parts/) yoki birlashtirilayotgan <fayl>.part.
"""
import os
import time
from pathlib import Path

MODELS = Path(os.getenv("COMFY_MODELS", str(Path.home() / "ComfyUI" / "models")))

FILES = [
    ("Asosiy model (Qwen-Image-Edit Q3_K_M)", "unet/qwen-image-edit-2511-Q3_K_M.gguf", 9920805472),
    ("Matn enkoderi (Qwen2.5-VL-7B)", "text_encoders/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", 4683072384),
    ("Rasm enkoderi (mmproj)", "text_encoders/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf", 1354163040),
    ("VAE", "vae/qwen_image_vae.safetensors", 253806246),
    ("Lightning LoRA (4 qadam)", "loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", 849608296),
]


def _dir_size(p: Path) -> int:
    try:
        return sum(f.stat().st_size for f in p.iterdir() if f.is_file())
    except FileNotFoundError:
        return 0


def status() -> dict:
    files = []
    for name, rel, size in FILES:
        path = MODELS / rel
        parts = Path(str(path) + ".parts")
        part = Path(str(path) + ".part")
        if path.exists():
            state, done = "done", size
        elif parts.exists():
            state, done = "downloading", _dir_size(parts)
            if part.exists():  # bo'laklar bitta faylga birlashtirilmoqda
                state, done = "merging", size
        elif part.exists():
            state, done = "merging", size
        else:
            state, done = "pending", 0
        files.append({"name": name, "state": state, "done": min(done, size), "size": size})
    total = sum(f["size"] for f in files)
    done = sum(f["done"] for f in files)
    return {
        "complete": all(f["state"] == "done" for f in files),
        "done": done,
        "total": total,
        "files": files,
        "time": time.time(),
    }
