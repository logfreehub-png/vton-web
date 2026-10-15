"""FastAPI server: API endpointlar + interfeys (static/index.html).

Ishga tushirish:  bash run.sh
"""
import base64
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from . import config
from .model import model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vton")

CATEGORIES = {"tops", "bottoms", "one-pieces"}
PHOTO_TYPES = {"model", "flat-lay"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Server yonganda model GPU'ga bir marta yuklanadi
    model.load()
    if config.SAVE_OUTPUTS:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="VTON test", lifespan=lifespan)


async def read_image(upload: UploadFile, label: str) -> Image.Image:
    """Yuklangan faylni tekshirib, RGB rasmga aylantiradi."""
    data = await upload.read()
    if not data:
        raise HTTPException(400, f"{label} rasmi bo'sh")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{label} rasmi {config.MAX_UPLOAD_MB} MB dan katta")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # telefon rasmlari ag'darilib qolmasin
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"{label}: rasm formati tanilmadi (JPG, PNG yoki WEBP yuboring)")


def to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/api/health")
def health():
    """Server va model holati."""
    return {
        "ready": model.ready,
        "device": model.device_name,
        "load_seconds": round(model.load_seconds, 1),
        "busy": model.busy,
        "waiting": model.waiting,
    }


@app.post("/api/tryon")
async def tryon(
    person: UploadFile = File(..., description="Odam rasmi"),
    garment: UploadFile = File(..., description="Kiyim rasmi"),
    category: str = Form("tops"),
    garment_photo_type: str = Form("model"),
    num_timesteps: int = Form(30),
    num_samples: int = Form(1),
    guidance_scale: float = Form(1.5),
    seed: int = Form(42),
    segmentation_free: bool = Form(True),
):
    # --- Parametrlarni tekshirish ---
    if category not in CATEGORIES:
        raise HTTPException(400, f"category noto'g'ri: {category}")
    if garment_photo_type not in PHOTO_TYPES:
        raise HTTPException(400, f"garment_photo_type noto'g'ri: {garment_photo_type}")
    if not 10 <= num_timesteps <= 50:
        raise HTTPException(400, "num_timesteps 10 va 50 orasida bo'lishi kerak")
    if not 1 <= num_samples <= 4:
        raise HTTPException(400, "num_samples 1 va 4 orasida bo'lishi kerak")
    if not 0.0 <= guidance_scale <= 5.0:
        raise HTTPException(400, "guidance_scale 0 va 5 orasida bo'lishi kerak")

    person_img = await read_image(person, "Odam")
    garment_img = await read_image(garment, "Kiyim")

    # --- Modelni ishga tushirish (alohida thread'da, server qotib qolmasligi uchun) ---
    t0 = time.perf_counter()
    try:
        images = await run_in_threadpool(
            model.run,
            person_image=person_img,
            garment_image=garment_img,
            category=category,
            garment_photo_type=garment_photo_type,
            num_samples=num_samples,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            seed=seed,
            segmentation_free=segmentation_free,
        )
    except Exception as e:  # GPU xotirasi to'lishi va boshqa xatolar
        log.exception("Inference xatosi")
        msg = str(e)
        if "out of memory" in msg.lower():
            msg = "GPU xotirasi yetmadi. num_samples ni kamaytiring yoki boshqa jarayonlarni to'xtating."
        raise HTTPException(500, f"Model xatosi: {msg}")
    seconds = time.perf_counter() - t0

    # --- Natijani diskka saqlash (ixtiyoriy) ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    if config.SAVE_OUTPUTS:
        folder = config.OUTPUT_DIR / run_id
        folder.mkdir(parents=True, exist_ok=True)
        person_img.save(folder / "person.jpg", quality=92)
        garment_img.save(folder / "garment.jpg", quality=92)
        for i, img in enumerate(images):
            img.save(folder / f"result_{i:02d}.png")

    log.info("Try-on tayyor: %s, %.2f s, %s, %d ta rasm", run_id, seconds, category, len(images))
    return {
        "id": run_id,
        "seconds": round(seconds, 2),
        "device": model.device_name,
        "images": [to_base64_png(img) for img in images],
    }


# Interfeys: http://SERVER_IP:8000/ -> static/index.html
# (API yo'llaridan KEYIN ulanadi, aks holda /api/* ni yopib qo'yadi)
app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
