"""
recollect_metrics.py
────────────────────
Re-runs local inference for all 50 prompts to collect CORRECT hardware
metrics (latency, peak RAM, avg CPU) using the fixed hardware_monitor.py.

Does NOT re-judge. Does NOT touch rq1_scores.csv or rq1_responses.csv.
Only overwrites rq2_metrics.csv with accurate numbers.

Why this script exists:
    The original hardware_monitor measured the Python process RAM (~40 MB),
    not the Ollama subprocess where the model actually lives. This script
    fixes that by re-running inference with the corrected monitor.

Run:
    taskpolicy -b python recollect_metrics.py
"""

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

from mt_bench_prompts import PROMPTS
from hardware_monitor import HardwareMonitor

# ── Same config as run_experiments.py ────────────────────────────────────────

LOCAL_MODELS = [
    {"name": "Qwen2.5-0.5B", "ollama_id": "qwen2.5:0.5b-instruct-fp16"},
    {"name": "Llama-3.2-1B", "ollama_id": "llama3.2:1b-instruct-fp16"},
]

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

METRICS_CSV   = RESULTS_DIR / "rq2_metrics.csv"
METRICS_FIELDS = ["prompt_id", "category", "model",
                  "latency_s", "peak_ram_mb", "avg_cpu_pct"]

_CSV_KEY = ("prompt_id", "model")

_SEP  = "═" * 62
_DASH = "─" * 62


# ── Helpers ───────────────────────────────────────────────────────────────────

def _upsert_csv(path: Path, fields: list, row: dict):
    rows: list[dict] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    key = {k: str(row[k]) for k in _CSV_KEY}
    replaced = False
    for i, r in enumerate(rows):
        if all(str(r.get(k, "")) == key[k] for k in _CSV_KEY):
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def run_ollama(ollama_id: str, prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model":   ollama_id,
            "prompt":  prompt,
            "stream":  False,
            "options": {"num_thread": 4},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def unload_ollama(ollama_id: str):
    try:
        requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": ollama_id, "keep_alive": 0},
            timeout=15,
        )
    except Exception:
        pass
    time.sleep(3)


def _bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total)
    pct    = int(100 * done / total)
    return f"{'█' * filled}{'░' * (width - filled)}  {pct}%  ({done}/{total})"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    total = len(PROMPTS) * len(LOCAL_MODELS)
    done  = 0

    print(_SEP)
    print(f"  Metrics recollection started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Prompts : {len(PROMPTS)}  ×  Models : {len(LOCAL_MODELS)}  =  {total} inferences")
    print(f"  Output  : {METRICS_CSV}")
    print(_SEP)

    for p in PROMPTS:
        pid = p["id"]
        cat = p["category"]
        q   = p["text"]

        for m in LOCAL_MODELS:
            mname = m["name"]
            oid   = m["ollama_id"]

            print(f"\n  Prompt {pid:>2}/{len(PROMPTS)}  {mname}")

            monitor = HardwareMonitor()
            try:
                monitor.start()
                run_ollama(oid, q)
                hw = monitor.stop()
                print(f"    Latency : {hw['latency_s']}s"
                      f"  │  Peak RAM : {hw['peak_ram_mb']} MB"
                      f"  │  Avg CPU  : {hw['avg_cpu_pct']}%")
                _upsert_csv(METRICS_CSV, METRICS_FIELDS, {
                    "prompt_id":   pid,
                    "category":    cat,
                    "model":       mname,
                    "latency_s":   hw["latency_s"],
                    "peak_ram_mb": hw["peak_ram_mb"],
                    "avg_cpu_pct": hw["avg_cpu_pct"],
                })
                done += 1
            except Exception as e:
                monitor.stop()
                print(f"    ERROR: {e}")
            finally:
                unload_ollama(oid)

            print(f"  Progress : {_bar(done, total)}")
            time.sleep(2)

    print()
    print(_SEP)
    print(f"  Done : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {done}/{total} metrics collected → {METRICS_CSV}")
    print(f"  Next step: python analysis_rq1.py")
    print(_SEP)


if __name__ == "__main__":
    if not os.getenv("OLLAMA_HOST", ""):
        pass  # OLLAMA_HOST has a default, this is fine
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted — re-run to continue (CSV already has partial data).")
        sys.exit(0)
