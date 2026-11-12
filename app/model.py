"""Qwen-Image-Edit-2511 (GGUF) bilan ishlash — model ComfyUI ichida turadi.

ComfyUI alohida servis bo'lib 127.0.0.1:8188 da ishlaydi va modelni GPU'da ushlab turadi.
Bu fayl unga rasmlarni yuklaydi, workflow yuboradi va natijani olib keladi.
GPU bitta, shuning uchun so'rovlar navbat bilan ishlaydi (threading.Lock).
"""
import io
import json
import logging
import threading
import time
import urllib.parse
import urllib.request
import uuid

from PIL import Image

from . import config

log = logging.getLogger("vton")

# Do'kon kiyim rasmida nimalar borligini belgilaydi. Promptda FAQAT belgilangan narsalar nomlanadi:
# bu model gapda biror narsa nomi (kepka, cho'ntak, pidjak...) bo'lsa, rasmda bo'lmasa ham uni
# o'zidan qo'shib yuboradi. Shuning uchun promptda "qo'shma" deb ham nom tilga olinmaydi.
ITEMS = {
    "top": "the top",
    "bottom": "the trousers or skirt",
    "dress": "the dress or jumpsuit",
    "outer": "the jacket or coat (worn over the top)",
    "shoes": "the shoes",
    "headwear": "the cap or hat (worn on the head)",
    "eyewear": "the glasses (worn on the face)",
    "watch": "the watch (worn on the wrist)",
    "bag": "the bag",
}
ITEM_ORDER = list(ITEMS)

# Rasmda bo'lmagan asosiy kiyim — odamniki o'zgarmay qoladi
KEEP = {
    "top": "top",
    "bottom": "trousers",
    "shoes": "shoes",
}

# Bu shablon sinovlarda eng barqaror ishladi: model 1-rasmni (odamni) tahrirlaydi.
# Promptning boshini "Edit image 1 ..." ga o'zgartirish yoki kiyimlarni mavhum nomlash
# (upper-body garment) modelni 2-rasmni tahrirlashga olib keldi — o'zgartirmang.
PROMPT_TEMPLATE = (
    "Image 2 is a store product photo of the items for sale. Dress the person in image 1 in exactly "
    "the items that are visible in image 2 and nothing else. Image 2 contains {items}; the person must "
    "wear {all_of}. "
    "Each item must look exactly as in image 2: same color and shade, material, pattern, shape, length, "
    "trims, stripes, logos and details. Light colors stay light. "
    "Do not invent any extra items, layers, pockets, patches or text. "
    "Each item from image 2 replaces the person's current item of the same type. "
    "{keep}"
    "Do not change the person: keep the exact same face, identity, facial expression, skin tone, hair, "
    "body shape and proportions, pose, hands and the background as in image 1. "
    "Only the clothing changes. Photorealistic."
)


# items=auto: rasmdagi hamma narsa, hech bir narsa nomi tilga olinmaydi (sinovda eng barqaror).
# Do'kon platformasi (try-on-clothes) shu rejimdan foydalanadi.
AUTO_TEMPLATE = (
    "Image 2 is a store product photo of the items for sale. Dress the person in image 1 in exactly "
    "the items that are visible in image 2 and nothing else: every item in image 2 must be worn, "
    "and no item that is absent from image 2 may be added. "
    "Each item must look exactly as in image 2: same color and shade, material, pattern, shape, length, "
    "trims, stripes, logos and details. Light colors stay light. "
    "Do not invent any extra items, layers, pockets, patches or text. "
    "Each garment from image 2 replaces the person's current garment of the same type; "
    "any part of the person's current outfit that has no replacement in image 2 stays unchanged. "
    "Do not change the person: keep the exact same face, identity, facial expression, skin tone, hair, "
    "body shape and proportions, pose, hands and the background as in image 1. "
    "Only the clothing changes. Photorealistic."
)


def build_prompt(items: list, extra: str = "") -> str:
    if "auto" in items:
        return AUTO_TEMPLATE + (" " + extra.strip() if extra.strip() else "")
    items = [i for i in ITEM_ORDER if i in items]
    n = len(items)
    names = [ITEMS[i] for i in items]
    listed = names[0] if n == 1 else ", ".join(names[:-1]) + " and " + names[-1]

    covered = set(items)
    if "dress" in covered:
        covered |= {"top", "bottom"}
    keep = "".join(f"The person keeps their own {name} from image 1 unchanged. "
                   for key, name in KEEP.items() if key not in covered)

    prompt = PROMPT_TEMPLATE.format(items=listed, all_of="it" if n == 1 else f"all {n} of them", keep=keep)
    if extra.strip():
        prompt += " " + extra.strip()
    return prompt


def build_workflow(person_name, garment_name, prompt, seed, steps):
    c = config
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": c.UNET}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": c.LORA, "strength_model": 1.0}},
        "3": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["2", 0], "shift": 3.1}},
        "5": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": c.TEXT_ENCODER, "type": "qwen_image"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": c.VAE}},
        "7": {"class_type": "LoadImage", "inputs": {"image": person_name}},
        "8": {"class_type": "LoadImage", "inputs": {"image": garment_name}},
        "9": {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"image": ["7", 0], "upscale_method": "lanczos",
                         "megapixels": c.MEGAPIXELS, "resolution_steps": 16}},
        "10": {"class_type": "TextEncodeQwenImageEditPlus",
               "inputs": {"clip": ["5", 0], "prompt": prompt, "vae": ["6", 0],
                          "image1": ["9", 0], "image2": ["8", 0]}},
        "12": {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": ["10", 0], "reference_latents_method": "index_timestep_zero"}},
        # cfg=1 bo'lgani uchun negative ishlatilmaydi: rasmlarni qayta kodlamay, nol shart beramiz
        "13": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["12", 0]}},
        "14": {"class_type": "VAEEncode", "inputs": {"pixels": ["9", 0], "vae": ["6", 0]}},
        "15": {"class_type": "KSampler",
               "inputs": {"model": ["3", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple",
                          "positive": ["12", 0], "negative": ["13", 0],
                          "latent_image": ["14", 0], "denoise": 1.0}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["6", 0]}},
        "17": {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": "vton"}},
    }


class ComfyClient:
    def __init__(self, base):
        self.base = base.rstrip("/")

    def _get(self, path, timeout=10):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return r.read()

    def _post_json(self, path, data, timeout=30):
        req = urllib.request.Request(self.base + path, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI xatosi: {e.read().decode(errors='replace')[:500]}")

    def upload(self, img: Image.Image, name: str) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + buf.getvalue() + (
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue"
            f"\r\n--{boundary}--\r\n").encode()
        req = urllib.request.Request(self.base + "/upload/image", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["name"]

    def run(self, workflow, timeout=1800) -> list:
        pid = self._post_json("/prompt", {"prompt": workflow, "client_id": "vton-web"})["prompt_id"]
        t0 = time.time()
        while time.time() - t0 < timeout:
            hist = json.loads(self._get(f"/history/{pid}"))
            if pid in hist:
                h = hist[pid]
                status = h.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    detail = msgs[0][1].get("exception_message", "") if msgs else ""
                    raise RuntimeError(f"ComfyUI ishlashda xato: {detail}")
                images = []
                for out in h.get("outputs", {}).values():
                    for im in out.get("images", []):
                        q = urllib.parse.urlencode(im)
                        images.append(Image.open(io.BytesIO(self._get(f"/view?{q}", timeout=60))).convert("RGB"))
                return images
            time.sleep(1)
        raise RuntimeError("ComfyUI javob bermadi (timeout)")


class VtonModel:
    def __init__(self):
        self.client = ComfyClient(config.COMFY_URL)
        self.device_name = "noma'lum"
        self.load_seconds = 0.0
        self._lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self.waiting = 0
        self.busy = False

    @property
    def ready(self) -> bool:
        try:
            stats = json.loads(self.client._get("/system_stats", timeout=3))
            # masalan: "cuda:0 NVIDIA GeForce RTX 3060 : cudaMallocAsync"
            gpu = stats["devices"][0]["name"].split(" : ")[0].split(" ", 1)[-1]
            self.device_name = "Qwen-Image-Edit-2511 · " + gpu
            return True
        except Exception:
            return False

    def load(self) -> None:
        """ComfyUI'ni tekshiradi; model birinchi so'rovda GPU'ga yuklanadi."""
        log.info("ComfyUI: %s, holati: %s", config.COMFY_URL, "tayyor" if self.ready else "ulanmadi")

    def run(self, person_image, garment_image, items, num_samples=1,
            num_timesteps=4, seed=42, extra_prompt="", **_) -> list:
        prompt = build_prompt(items, extra_prompt)
        log.info("Prompt: %s", prompt)
        with self._counter_lock:
            self.waiting += 1
        with self._lock:
            with self._counter_lock:
                self.waiting -= 1
            self.busy = True
            try:
                tag = uuid.uuid4().hex[:8]
                p = self.client.upload(person_image, f"person_{tag}.png")
                g = self.client.upload(garment_image, f"garment_{tag}.png")
                images = []
                try:
                    for i in range(num_samples):
                        wf = build_workflow(p, g, prompt, seed + i, num_timesteps)
                        images += self.client.run(wf)
                finally:
                    # disk to'lib qolmasin: ComfyUI'ga yuklangan va u saqlagan fayllarni o'chiramiz
                    for d, pat in ((config.COMFY_DIR / "input", f"*_{tag}.png"), (config.COMFY_DIR / "output", "vton_*.png")):
                        for f in d.glob(pat):
                            f.unlink(missing_ok=True)
                return images
            finally:
                self.busy = False


model = VtonModel()
