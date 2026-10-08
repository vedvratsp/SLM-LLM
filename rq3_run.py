"""
rq3_run.py — Run the 6 ethics-approved custom prompts through all models.

Produces:
  results/rq3_responses.csv  — raw responses for the 6 custom prompts
  results/rq3_scores.csv     — judge scores for those responses
  results/rq3_metrics.csv    — hardware metrics (local models)

Run AFTER run_experiments.py (which covers all 80 MT-Bench prompts).
Then run rq3_prep.py to build the A/B survey file.

The 6 custom prompts are in rq3_prompts.py.
"""

import csv, json, os, sys, time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

from rq3_prompts import RQ3_PROMPTS

# Re-use helpers from run_experiments (same file, same dir)
from run_experiments import (
    LOCAL_MODELS, OLLAMA_HOST, RESULTS_DIR,
    run_ollama, run_gemini, judge_response,
    _response_row, _score_row, _metric_row, _save_csv,
)
from hardware_monitor import HardwareMonitor


def run_rq3():
    responses = []
    scores    = []
    metrics   = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = RESULTS_DIR / f"rq3_log_{timestamp}.txt"

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"RQ3 experiment started: {datetime.now().isoformat()}")
    log("=" * 60)

    for p in RQ3_PROMPTS:
        pid = p["id"]
        cat = p["category"]
        q   = p["text"]

        log(f"\n── RQ3 Prompt {pid}/6: {cat} ──")
        log(f"  {q[:80]}...")

        # LOCAL MODELS
        for m in LOCAL_MODELS:
            mname = m["name"]
            oid   = m["ollama_id"]
            log(f"\n  [{mname}] running...")

            monitor = HardwareMonitor()
            try:
                monitor.start()
                resp = run_ollama(oid, q)
                hw   = monitor.stop()
            except Exception as e:
                monitor.stop()
                log(f"  ✗ Ollama error for {mname}: {e}")
                continue

            log(f"  ✓ {len(resp)} chars | latency={hw['latency_s']}s "
                f"RAM={hw['peak_ram_mb']}MB CPU={hw['avg_cpu_pct']}%")

            sc = judge_response(q, resp)
            log(f"  Scores → {sc}")

            responses.append(_response_row(pid, cat, mname, resp, "local"))
            scores.append(_score_row(pid, cat, mname, sc))
            metrics.append(_metric_row(pid, cat, mname, hw))
            time.sleep(1)

        # GEMINI
        log(f"\n  [Gemini 3.8 Flash] running...")
        try:
            resp_g = run_gemini(q)
        except Exception as e:
            log(f"  ✗ Gemini error: {e}")
            continue

        log(f"  ✓ {len(resp_g)} chars")
        sc_g = judge_response(q, resp_g)
        log(f"  Scores → {sc_g}")

        responses.append(_response_row(pid, cat, "Gemini 3.8 Flash", resp_g, "cloud"))
        scores.append(_score_row(pid, cat, "Gemini 3.8 Flash", sc_g))
        time.sleep(1)

    _save_csv(RESULTS_DIR / "rq3_responses.csv", responses,
              ["prompt_id", "category", "model", "source", "response"])
    _save_csv(RESULTS_DIR / "rq3_scores.csv", scores,
              ["prompt_id", "category", "model", "accuracy", "clarity", "completeness", "overall"])
    _save_csv(RESULTS_DIR / "rq3_metrics.csv", metrics,
              ["prompt_id", "category", "model", "latency_s", "peak_ram_mb", "avg_cpu_pct"])

    log("\nDone. Run  python rq3_prep.py  next to build the A/B survey file.")


if __name__ == "__main__":
    missing = [v for v in ("GROQ_API_KEY", "GEMINI_API_KEY") if not os.getenv(v)]
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}")
        sys.exit(1)
    run_rq3()
