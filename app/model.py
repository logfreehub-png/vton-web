"""Modelni bir marta yuklash va GPU'da xavfsiz ishlatish.

GPU bitta, model bitta. Bir vaqtda 2 ta so'rov kelsa, ular navbat bilan
ishlaydi (threading.Lock). Aks holda GPU xotirasi to'lib qolishi mumkin.
"""
import logging
import threading
import time

from . import config

log = logging.getLogger("vton")


class VtonModel:
    def __init__(self):
        self.pipeline = None
        self.device_name = "noma'lum"
        self.load_seconds = 0.0
        self._lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self.waiting = 0  # navbatda turgan so'rovlar soni
        self.busy = False  # hozir GPU ishlayaptimi

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def load(self) -> None:
        """Server ishga tushganda bir marta chaqiriladi."""
        from fashn_vton import TryOnPipeline

        log.info("Model yuklanmoqda: %s", config.WEIGHTS_DIR)
        t0 = time.perf_counter()
        self.pipeline = TryOnPipeline(weights_dir=config.WEIGHTS_DIR, device=config.DEVICE)
        self.load_seconds = time.perf_counter() - t0

        try:
            import torch

            if torch.cuda.is_available():
                self.device_name = torch.cuda.get_device_name(0)
            else:
                self.device_name = "CPU (GPU topilmadi)"
        except ImportError:
            self.device_name = "CPU"

        log.info("Model tayyor: %.1f s, qurilma: %s", self.load_seconds, self.device_name)

    def run(self, **kwargs) -> list:
        """Try-on'ni ishga tushiradi va PIL rasmlar ro'yxatini qaytaradi."""
        with self._counter_lock:
            self.waiting += 1
        with self._lock:  # oldingi so'rov tugashini kutadi
            with self._counter_lock:
                self.waiting -= 1
            self.busy = True
            try:
                result = self.pipeline(**kwargs)
            finally:
                self.busy = False
        return list(result.images)


# Butun ilova uchun yagona model obyekti
model = VtonModel()
