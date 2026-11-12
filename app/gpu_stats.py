"""GPU ko'rsatkichlari nvidia-smi orqali (admin paneldagi "Server resurslari" sahifasi uchun).

nvidia-smi yo'q bo'lsa (CPU server yoki lokal sinov) bo'sh ro'yxat qaytadi.
Natija 1 soniya keshlanadi: health so'rovlari ko'p bo'lsa ham nvidia-smi kam chaqiriladi.
"""
import shutil
import subprocess
import threading
import time

_FIELDS = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
_CACHE_SECONDS = 1.0
_lock = threading.Lock()
_cached: tuple[float, list[dict]] = (0.0, [])


def _num(value: str):
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return None  # "[N/A]" va shunga o'xshashlar


def _query() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        gpus.append({
            "index": int(_num(parts[0]) or 0),
            "name": parts[1],
            "util_percent": _num(parts[2]),
            "memory_used_mb": _num(parts[3]),
            "memory_total_mb": _num(parts[4]),
            "temperature_c": _num(parts[5]),
            "power_w": _num(parts[6]),
        })
    return gpus


def gpu_stats() -> list[dict]:
    global _cached
    with _lock:
        ts, data = _cached
        if time.monotonic() - ts < _CACHE_SECONDS:
            return data
        data = _query()
        _cached = (time.monotonic(), data)
        return data
