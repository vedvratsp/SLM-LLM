"""
hardware_monitor.py — RQ1/RQ2 hardware metrics collector.

Measures wall-clock latency, peak RAM, and average CPU during a single
Ollama inference call.

RAM measurement strategy
────────────────────────
Queries Ollama's /api/ps endpoint every 0.5s during inference.
size_vram = bytes the model occupies in unified memory (Metal allocation).
On Apple Silicon, unified memory is shared CPU/GPU — this is the real number.
"""

import threading
import time
import psutil
import requests

OLLAMA_HOST = "http://localhost:11434"


def _ollama_ram_mb() -> float:
    """Query Ollama /api/ps and return total size_vram in MB."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=2)
        data = resp.json()
        total = sum(m.get("size_vram", 0) for m in data.get("models", []))
        return total / 1024 / 1024
    except Exception:
        return 0.0


class HardwareMonitor:
    SAMPLE_INTERVAL = 0.5

    def __init__(self):
        self._cpu     = []
        self._ram     = []
        self._running = False
        self._thread  = None
        self._t0      = None

    def _sample(self):
        while self._running:
            self._cpu.append(psutil.cpu_percent(interval=None))
            self._ram.append(_ollama_ram_mb())
            time.sleep(self.SAMPLE_INTERVAL)

    def start(self):
        self._cpu     = []
        self._ram     = []
        self._running = True
        self._t0      = time.perf_counter()
        self._thread  = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        latency = time.perf_counter() - self._t0
        self._running = False
        self._thread.join(timeout=2)
        return {
            "latency_s":   round(latency, 3),
            "peak_ram_mb": round(max(self._ram), 1) if self._ram else 0.0,
            "avg_cpu_pct": round(sum(self._cpu) / len(self._cpu), 1) if self._cpu else 0.0,
        }