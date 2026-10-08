"""
run_experiments.py
RQ1 (response quality) + RQ2 (hardware metrics) — production-grade

Architecture
────────────
• Checkpoint  results/checkpoint.json  tracks every (prompt_id, model) as
  "pending" | "success" | "error".
  A prompt advances ONLY when all three models reach "success".
  Errors are retried on the next run — re-run the script as many times as
  needed until all prompts are complete.

• Every CSV row is written (or overwritten) immediately after a successful
  inference + judging pair.  Writes are atomic (tmp-file rename) so a
  hard kill never corrupts a CSV.

• Ollama is explicitly told to release the model from memory (keep_alive=0)
  after every inference call, followed by a 3-second pause so macOS can
  reclaim the RAM before loading the next model.

• Log is unambiguous: the judging model is always named separately from the
  model being judged.

Models
──────
  Local  : Qwen2.5-0.5B  (FP16, ~1 GB)  via Ollama (localhost)
           Llama-3.2-1B  (FP16, ~2 GB)  via Ollama (localhost)
  Cloud  : GPT-OSS-120B                  via Groq API
  Judge  : Qwen3.8-27B                   via Groq API

  Both local models run in full FP16 precision — zero quantization applied.
  This satisfies RQ1's constraint: unquantized inference on memory-constrained
  hardware (≤ 8 GB unified memory).

  Pull before first run:
      ollama pull qwen2.5:0.5b-instruct-fp16
      ollama pull llama3.2:1b-instruct-fp16

Run command (keeps Mac responsive)
───────────────────────────────────
  taskpolicy -b python run_experiments.py

Resume after crash — just re-run:
  python run_experiments.py
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

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_MODELS = [
    # Full FP16 — no quantization. Fits comfortably within ~5 GB free RAM
    # on an 8 GB device (macOS keeps ~3 GB for itself).
    {"name": "Qwen2.5-0.5B", "ollama_id": "qwen2.5:0.5b-instruct-fp16"},
    {"name": "Llama-3.2-1B", "ollama_id": "llama3.2:1b-instruct-fp16"},
]

CLOUD_MODEL_NAME    = "GPT-OSS-120B"
CLOUD_MODEL_GROQ_ID = "openai/gpt-oss-120b"

JUDGE_MODEL_ID    = "qwen/qwen3.8-27b"
JUDGE_MODEL_LABEL = "Groq Qwen3.8-27B"

ALL_MODEL_NAMES = [m["name"] for m in LOCAL_MODELS] + [CLOUD_MODEL_NAME]

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint  (pending | success | error per prompt+model)
# ─────────────────────────────────────────────────────────────────────────────

S_PENDING = "pending"
S_SUCCESS = "success"
S_ERROR   = "error"


def load_checkpoint() -> dict:
    """Load {str(prompt_id): {model_name: status}} from disk."""
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}   # corrupted file — start fresh


def _save_checkpoint(cp: dict):
    """Atomic write via tmp-file rename."""
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cp, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


def get_status(cp: dict, prompt_id, model: str) -> str:
    return cp.get(str(prompt_id), {}).get(model, S_PENDING)


def set_status(cp: dict, prompt_id, model: str, status: str):
    pid = str(prompt_id)
    if pid not in cp:
        cp[pid] = {m: S_PENDING for m in ALL_MODEL_NAMES}
    cp[pid][model] = status
    _save_checkpoint(cp)


def prompt_complete(cp: dict, prompt_id) -> bool:
    """True only when every model for this prompt is SUCCESS."""
    pid = str(prompt_id)
    return all(
        cp.get(pid, {}).get(m, S_PENDING) == S_SUCCESS
        for m in ALL_MODEL_NAMES
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV  (atomic upsert — crash-safe, rewrites errors on retry)
# ─────────────────────────────────────────────────────────────────────────────

RESPONSE_FIELDS = ["prompt_id", "category", "model", "source", "response"]
SCORE_FIELDS    = ["prompt_id", "category", "model",
                   "accuracy", "clarity", "completeness", "overall"]
METRIC_FIELDS   = ["prompt_id", "category", "model",
                   "latency_s", "peak_ram_mb", "avg_cpu_pct"]

RESPONSES_CSV = RESULTS_DIR / "rq1_responses.csv"
SCORES_CSV    = RESULTS_DIR / "rq1_scores.csv"
METRICS_CSV   = RESULTS_DIR / "rq2_metrics.csv"

_CSV_KEY = ("prompt_id", "model")   # uniqueness key for upsert


def _ensure_csv(path: Path, fields: list):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _upsert_csv(path: Path, fields: list, row: dict):
    """
    Insert or update a row matched by (prompt_id, model).
    Existing error entries are overwritten when the retry succeeds.
    Atomic via tmp-file rename.
    """
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


def persist(prompt_id, category, model_name, source,
            response, scores, metrics=None):
    """Write all three CSVs for one (prompt, model) result."""
    _upsert_csv(RESPONSES_CSV, RESPONSE_FIELDS, {
        "prompt_id": prompt_id, "category": category,
        "model": model_name, "source": source, "response": response,
    })
    _upsert_csv(SCORES_CSV, SCORE_FIELDS, {
        "prompt_id": prompt_id, "category": category,
        "model": model_name, **scores,
    })
    if metrics:
        _upsert_csv(METRICS_CSV, METRIC_FIELDS, {
            "prompt_id": prompt_id, "category": category,
            "model": model_name, **metrics,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Ollama  (inference + forced memory release)
# ─────────────────────────────────────────────────────────────────────────────

def run_ollama(ollama_id: str, prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model":   ollama_id,
            "prompt":  prompt,
            "stream":  False,
            "options": {"num_thread": 4},  # leave performance cores free for macOS
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def unload_ollama(ollama_id: str):
    """
    Tell Ollama to evict the model from memory immediately.
    keep_alive=0 releases GPU/RAM on Apple Silicon.
    The 3-second sleep gives macOS time to reclaim pages before the
    next model loads — this is what keeps the Mac responsive.
    """
    try:
        requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": ollama_id, "keep_alive": 0},
            timeout=15,
        )
    except Exception:
        pass
    time.sleep(3)


# ─────────────────────────────────────────────────────────────────────────────
# Cloud inference  (Groq, exponential backoff)
# ─────────────────────────────────────────────────────────────────────────────

def run_cloud(prompt: str, log) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    for attempt in range(1, 5):
        try:
            resp = client.chat.completions.create(
                model=CLOUD_MODEL_GROQ_ID,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if attempt == 4:
                raise RuntimeError(f"Cloud model failed after 4 attempts: {e}")
            wait = (30 * attempt) if ("429" in err or "rate" in err.lower()) else (10 * attempt)
            log(f"    ⚠  Cloud attempt {attempt}/4 failed: {e} — retry in {wait}s")
            time.sleep(wait)


# ─────────────────────────────────────────────────────────────────────────────
# Judge  (Groq Qwen3.8-27B)
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an expert, impartial evaluator of AI-generated text. "
    "Score the response on three dimensions, each from 1 (very poor) to 5 (excellent). "
    "Reply ONLY with valid JSON — no explanation, no markdown, no extra text."
)

_JUDGE_TEMPLATE = """\
Question asked:
{question}

AI response to evaluate:
{response}

Score on:
- accuracy: factual correctness and appropriateness (1-5)
- clarity: readability and coherence (1-5)
- completeness: how fully the question is answered (1-5)

Return ONLY this JSON (replace X/Y/Z with integers 1-5):
{{"accuracy": X, "clarity": Y, "completeness": Z}}"""


def judge_response(question: str, response: str, log) -> dict:
    """
    Score one response via Groq Qwen3.8-27B.
    Raises RuntimeError after 3 failed attempts so the caller can mark
    the checkpoint as error and retry next run.
    """
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    snippet = response[:3000]

    for attempt in range(1, 4):
        try:
            msg = client.chat.completions.create(
                model=JUDGE_MODEL_ID,
                max_tokens=400,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user",   "content": _JUDGE_TEMPLATE.format(
                        question=question, response=snippet)},
                ],
            )
            raw = msg.choices[0].message.content.strip()
            scores = json.loads(raw)
            for k in ("accuracy", "clarity", "completeness"):
                scores[k] = max(1, min(5, int(scores[k])))
            scores["overall"] = round(
                sum(scores[k] for k in ("accuracy", "clarity", "completeness")) / 3, 2
            )
            return scores
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"Judge failed after 3 attempts: {e}")
            log(f"    ⚠  Judge attempt {attempt}/3 failed: {e} — retry in {attempt * 10}s")
            time.sleep(attempt * 10)


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

_SEP  = "═" * 62
_DASH = "─" * 62


def _bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total)
    pct    = int(100 * done / total)
    return f"{'█' * filled}{'░' * (width - filled)}  {pct}%  ({done}/{total} prompts)"


def make_logger(log_path: Path):
    def log(msg: str = ""):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    return log


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    _ensure_csv(RESPONSES_CSV, RESPONSE_FIELDS)
    _ensure_csv(SCORES_CSV,    SCORE_FIELDS)
    _ensure_csv(METRICS_CSV,   METRIC_FIELDS)

    cp = load_checkpoint()
    done_count = sum(1 for p in PROMPTS if prompt_complete(cp, p["id"]))

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = make_logger(RESULTS_DIR / f"run_log_{ts}.txt")

    log(_SEP)
    log(f"  Experiment started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Total prompts       : {len(PROMPTS)}")
    log(f"  Already complete    : {done_count}")
    log(f"  Remaining           : {len(PROMPTS) - done_count}")
    log(f"  Local models        : {', '.join(m['name'] for m in LOCAL_MODELS)} (FP16, unquantized)")
    log(f"  Cloud model         : {CLOUD_MODEL_NAME}  (Groq)")
    log(f"  Judge               : {JUDGE_MODEL_LABEL}  (Groq)")
    log(_SEP)

    for p in PROMPTS:
        pid = p["id"]
        cat = p["category"]
        q   = p["text"]

        if prompt_complete(cp, pid):
            log(f"  [SKIP] Prompt {pid:>2}/{len(PROMPTS)} ({cat}) — all models complete.")
            continue

        log()
        log(_SEP)
        log(f"  Prompt {pid}/{len(PROMPTS)}  │  {cat}  │  {datetime.now().strftime('%H:%M:%S')}")
        log(_SEP)
        log(f"  Q: {q[:115]}{'…' if len(q) > 115 else ''}")

        prompt_ok = True

        # ── LOCAL MODELS ─────────────────────────────────────────────────────
        for idx, m in enumerate(LOCAL_MODELS, start=1):
            mname = m["name"]
            oid   = m["ollama_id"]

            if get_status(cp, pid, mname) == S_SUCCESS:
                log(f"\n  [SKIP] {mname} already succeeded for prompt {pid}.")
                continue

            log()
            log(f"  ▶ [{idx}/{len(ALL_MODEL_NAMES)}] {mname} — local inference (FP16)")

            monitor = HardwareMonitor()
            resp = hw = None

            try:
                monitor.start()
                resp = run_ollama(oid, q)
                hw   = monitor.stop()
                log(f"    Status   : ✓ completed")
                log(f"    Response : {len(resp)} chars")
                log(f"    Latency  : {hw['latency_s']}s"
                    f"  │  Peak RAM : {hw['peak_ram_mb']} MB"
                    f"  │  Avg CPU  : {hw['avg_cpu_pct']}%")
            except Exception as e:
                monitor.stop()
                log(f"    Status   : ✗ inference error — {e}")
                set_status(cp, pid, mname, S_ERROR)
                prompt_ok = False
            finally:
                log(f"    Memory   : unloading {mname} from Ollama…")
                unload_ollama(oid)

            if resp is None:
                continue

            log(f"    Judge    : {JUDGE_MODEL_LABEL}")
            try:
                sc = judge_response(q, resp, log)
                log(f"    Scores   : Accuracy={sc['accuracy']}"
                    f"  Clarity={sc['clarity']}"
                    f"  Completeness={sc['completeness']}"
                    f"  Overall={sc['overall']}")
                persist(pid, cat, mname, "local", resp, sc, hw)
                set_status(cp, pid, mname, S_SUCCESS)
            except Exception as e:
                log(f"    Judge    : ✗ failed — {e}")
                set_status(cp, pid, mname, S_ERROR)
                prompt_ok = False

            time.sleep(2)

        # ── CLOUD MODEL ───────────────────────────────────────────────────────
        if get_status(cp, pid, CLOUD_MODEL_NAME) == S_SUCCESS:
            log(f"\n  [SKIP] {CLOUD_MODEL_NAME} already succeeded for prompt {pid}.")
        else:
            log()
            log(f"  ▶ [{len(ALL_MODEL_NAMES)}/{len(ALL_MODEL_NAMES)}]"
                f" {CLOUD_MODEL_NAME} — cloud inference (Groq)")
            resp_c = None
            try:
                resp_c = run_cloud(q, log)
                log(f"    Status   : ✓ completed")
                log(f"    Response : {len(resp_c)} chars")
            except Exception as e:
                log(f"    Status   : ✗ inference error — {e}")
                set_status(cp, pid, CLOUD_MODEL_NAME, S_ERROR)
                prompt_ok = False

            if resp_c is not None:
                log(f"    Judge    : {JUDGE_MODEL_LABEL}")
                try:
                    sc_c = judge_response(q, resp_c, log)
                    log(f"    Scores   : Accuracy={sc_c['accuracy']}"
                        f"  Clarity={sc_c['clarity']}"
                        f"  Completeness={sc_c['completeness']}"
                        f"  Overall={sc_c['overall']}")
                    persist(pid, cat, CLOUD_MODEL_NAME, "cloud", resp_c, sc_c)
                    set_status(cp, pid, CLOUD_MODEL_NAME, S_SUCCESS)
                except Exception as e:
                    log(f"    Judge    : ✗ failed — {e}")
                    set_status(cp, pid, CLOUD_MODEL_NAME, S_ERROR)
                    prompt_ok = False

            time.sleep(2)

        # ── Prompt result ─────────────────────────────────────────────────────
        if prompt_ok and prompt_complete(cp, pid):
            done_count += 1
            log()
            log(f"  ✓ Prompt {pid} complete — all 3 models succeeded.")
        else:
            log()
            log(f"  ✗ Prompt {pid} incomplete — failed models will retry on next run.")

        log(f"  Progress : {_bar(done_count, len(PROMPTS))}")
        log(_DASH)

    # ── Final summary ─────────────────────────────────────────────────────────
    final = sum(1 for p in PROMPTS if prompt_complete(cp, p["id"]))
    log()
    log(_SEP)
    log(f"  Run finished    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Prompts done    : {final} / {len(PROMPTS)}")
    if final < len(PROMPTS):
        log(f"  ⚠  {len(PROMPTS) - final} prompt(s) still have errors.")
        log("     Re-run the script to retry them.")
        log("     Command:  taskpolicy -b python run_experiments.py")
    else:
        log("  ✓ All prompts complete!")
        log("     Next step:  python analysis_rq1.py")
    log(_SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Add it to your .env file.")
        sys.exit(1)
    try:
        run_all()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Checkpoint saved — re-run to continue from here.")
        sys.exit(0)
