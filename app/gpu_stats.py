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


# ---------------------------------------------------------------------------
# Server (host) ko'rsatkichlari: CPU, RAM, swap, disk, load, uptime.
# Admin paneldagi "Server resurslari" sahifasida GPU serverning o'zini ham ko'rsatish uchun.
_host_lock = threading.Lock()
_host_cached: tuple[float, dict | None] = (0.0, None)


def _host_query() -> dict | None:
    try:
        import os
        import psutil
    except ImportError:
        return None
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage("/")
    try:
        load = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        load = None
    return {
        # interval=None: oldingi chaqiruvdan beri o'rtacha (kesh 1 s, ya'ni ~1 soniyalik oyna)
        "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
        "cpu_count": psutil.cpu_count(logical=True),
        "load_avg": load,
        "memory": {"used": mem.total - mem.available, "total": mem.total, "percent": round(mem.percent, 1)},
        "swap": {"used": swap.used, "total": swap.total, "percent": round(swap.percent, 1)},
        "disk": {"used": disk.used, "total": disk.total, "percent": round(disk.percent, 1)},
        "uptime_seconds": int(time.time() - psutil.boot_time()),
    }


def host_stats() -> dict | None:
    """GPU serverning CPU / RAM / disk holati. psutil o'rnatilmagan bo'lsa None."""
    global _host_cached
    with _host_lock:
        ts, data = _host_cached
        if data is not None and time.monotonic() - ts < _CACHE_SECONDS:
            return data
        data = _host_query()
        _host_cached = (time.monotonic(), data)
        return data

