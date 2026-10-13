"""
run_rq2.py — RQ2: Augmenting Llama-3.2-1B with Quantization and RAG.

Ablation study — three variants, same 80 MT-Bench Turn-1 prompts:
──────────────────────────────────────────────────────────────────
  Variant A │ Llama-1B-Q4       │ Q4_K_M quantization, no RAG
  Variant B │ Llama-1B-FP16-RAG │ FP16 (baseline precision), RAG enabled
  Variant C │ Llama-1B-Q4-RAG   │ Q4_K_M + RAG  (full augmentation)

Reference baseline from RQ1:
  Llama-1B-FP16 (no RAG, no quantization) → see results/rq1_scores.csv

Design mirrors run_experiments.py exactly:
• Checkpoint: results/rq2_checkpoint.json — (prompt_id, variant) → status
• Atomic CSV writes (tmp-file rename) — crash-safe
• Same Groq Qwen3.8-27B judge, same scoring rubric
• Ollama keep_alive=0 + 3-second sleep between model calls
• Same logging format

Ollama pull before first run:
    ollama pull llama3.2:1b                   # Q4_K_M (Variants A and C)
    ollama pull llama3.2:1b-instruct-fp16     # FP16   (Variant B — already pulled for RQ1)

Run command:
    python build_rag_corpus.py        # once only
    taskpolicy -b python run_rq2.py   # keeps Mac responsive

Resume after crash:
    python run_rq2.py                 # checkpoint skips done items
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
from hardware_monitor  import HardwareMonitor
from rag_engine        import RAGEngine

# ─────────────────────────────────────────────────────────────────────────────
# Variant config
# ─────────────────────────────────────────────────────────────────────────────

VARIANTS = [
    {
        "name":       "Llama-1B-Q4",      # A — quantization only
        "ollama_id":  "llama3.2:1b",      # Ollama default = Q4_K_M (≈ Q4_0 footprint, better quality)
        "use_rag":    False,
        "label":      "Variant A: Q4_K_M, no RAG",
    },
    {
        "name":       "Llama-1B-FP16-RAG",        # B — RAG only
        "ollama_id":  "llama3.2:1b-instruct-fp16",
        "use_rag":    True,
        "label":      "Variant B: FP16, RAG",
    },
    {
        "name":       "Llama-1B-Q4-RAG",   # C — both
        "ollama_id":  "llama3.2:1b",       # same Q4_K_M tag
        "use_rag":    True,
        "label":      "Variant C: Q4_K_M + RAG",
    },
]

ALL_VARIANT_NAMES = [v["name"] for v in VARIANTS]

JUDGE_MODEL_ID    = "qwen/qwen3.8-27b"
JUDGE_MODEL_LABEL = "Groq Qwen3.8-27B"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR       = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SCORES_CSV        = RESULTS_DIR / "rq2_scores.csv"
METRICS_CSV       = RESULTS_DIR / "rq2_metrics.csv"
CHECKPOINT_PATH   = RESULTS_DIR / "rq2_checkpoint.json"

SCORE_FIELDS  = ["prompt_id", "category", "model", "accuracy", "clarity",
                 "completeness", "overall", "rag_used"]
METRIC_FIELDS = ["prompt_id", "category", "model", "latency_s",
                 "peak_ram_mb", "avg_cpu_pct", "rag_used"]

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

S_PENDING = "pending"
S_SUCCESS = "success"
S_ERROR   = "error"


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        return {}
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(cp: dict):
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cp, f)
    tmp.replace(CHECKPOINT_PATH)


def _cp_key(prompt_id: int, variant_name: str) -> str:
    return f"{prompt_id}|{variant_name}"


def get_status(cp: dict, prompt_id: int, variant_name: str) -> str:
    return cp.get(_cp_key(prompt_id, variant_name), S_PENDING)


def set_status(cp: dict, prompt_id: int, variant_name: str, status: str):
    cp[_cp_key(prompt_id, variant_name)] = status
    save_checkpoint(cp)


def prompt_complete(cp: dict, prompt_id: int) -> bool:
    return all(
        get_status(cp, prompt_id, v["name"]) == S_SUCCESS
        for v in VARIANTS
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers  (atomic upsert)
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_csv(path: Path, fields: list):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _upsert_csv(path: Path, fields: list, new_row: dict):
    """Read → replace-or-append → atomic write."""
    key_fields = ["prompt_id", "model"]

    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    key = {k: str(new_row[k]) for k in key_fields}
    replaced = False
    for i, row in enumerate(rows):
        if all(row[k] == key[k] for k in key_fields):
            rows[i] = {k: new_row.get(k, "") for k in fields}
            replaced = True
            break
    if not replaced:
        rows.append({k: new_row.get(k, "") for k in fields})

    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_ollama(ollama_id: str, prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model":      ollama_id,
            "prompt":     prompt,
            "stream":     False,
            "num_thread": 4,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def unload_ollama(ollama_id: str):
    """Release model from RAM; wait for macOS to reclaim memory pages."""
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
# Judge  (Groq Qwen3.8-27B) — identical to run_experiments.py
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
            raw    = msg.choices[0].message.content.strip()
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
    _ensure_csv(SCORES_CSV,  SCORE_FIELDS)
    _ensure_csv(METRICS_CSV, METRIC_FIELDS)

    cp         = load_checkpoint()
    done_count = sum(1 for p in PROMPTS if prompt_complete(cp, p["id"]))

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = make_logger(RESULTS_DIR / f"rq2_run_log_{ts}.txt")

    # Initialise RAG engine once (lazy-loads embedding model on first query)
    rag = RAGEngine()

    log(_SEP)
    log(f"  RQ2 Experiment started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Total prompts          : {len(PROMPTS)}")
    log(f"  Already complete       : {done_count}")
    log(f"  Remaining              : {len(PROMPTS) - done_count}")
    log(f"  Variants               :")
    for v in VARIANTS:
        log(f"    {v['name']:<22} | {v['label']}")
    log(f"  Judge                  : {JUDGE_MODEL_LABEL}  (Groq)")
    log(_SEP)

    for p in PROMPTS:
        pid = p["id"]
        cat = p["category"]
        q   = p["text"]

        if prompt_complete(cp, pid):
            log(f"  [SKIP] Prompt {pid:>3}/{len(PROMPTS)} ({cat}) — all variants complete.")
            continue

        log()
        log(_SEP)
        log(f"  Prompt {pid}/{len(PROMPTS)}  │  {cat}  │  {datetime.now().strftime('%H:%M:%S')}")
        log(_SEP)
        log(f"  Q: {q[:115]}{'…' if len(q) > 115 else ''}")

        prompt_ok = True

        for vi, v in enumerate(VARIANTS, start=1):
            vname  = v["name"]
            oid    = v["ollama_id"]
            use_rag = v["use_rag"]

            if get_status(cp, pid, vname) == S_SUCCESS:
                log(f"\n  [SKIP] {vname} already succeeded for prompt {pid}.")
                continue

            log()
            log(f"  ▶ [{vi}/{len(VARIANTS)}] {vname}  ({v['label']})")

            # Build prompt (augmented with RAG context if enabled)
            if use_rag:
                augmented_q = rag.augment(category=cat, question=q)
                rag_chars   = len(augmented_q) - len(q)
                log(f"    RAG      : ✓ context prepended (+{rag_chars} chars)")
            else:
                augmented_q = q
                log(f"    RAG      : disabled")

            monitor = HardwareMonitor()
            resp = hw = None

            try:
                monitor.start()
                resp = run_ollama(oid, augmented_q)
                hw   = monitor.stop()
                log(f"    Inference: ✓  {len(resp)} chars")
                log(f"    Latency  : {hw['latency_s']}s"
                    f"  │  Peak RAM : {hw['peak_ram_mb']} MB"
                    f"  │  Avg CPU  : {hw['avg_cpu_pct']}%")
            except Exception as e:
                monitor.stop()
                log(f"    Inference: ✗ error — {e}")
                set_status(cp, pid, vname, S_ERROR)
                prompt_ok = False
            finally:
                log(f"    Memory   : unloading {vname} from Ollama…")
                unload_ollama(oid)

            if resp is None:
                continue

            # Judge uses the *original* question so scores are comparable across
            # all variants (RAG context is not part of what's being evaluated)
            log(f"    Judge    : {JUDGE_MODEL_LABEL}")
            try:
                sc = judge_response(q, resp, log)
                log(f"    Scores   : Accuracy={sc['accuracy']}"
                    f"  Clarity={sc['clarity']}"
                    f"  Completeness={sc['completeness']}"
                    f"  Overall={sc['overall']}")

                _upsert_csv(SCORES_CSV, SCORE_FIELDS, {
                    "prompt_id":    pid,
                    "category":     cat,
                    "model":        vname,
                    "accuracy":     sc["accuracy"],
                    "clarity":      sc["clarity"],
                    "completeness": sc["completeness"],
                    "overall":      sc["overall"],
                    "rag_used":     use_rag,
                })
                _upsert_csv(METRICS_CSV, METRIC_FIELDS, {
                    "prompt_id":    pid,
                    "category":     cat,
                    "model":        vname,
                    "latency_s":    hw["latency_s"],
                    "peak_ram_mb":  hw["peak_ram_mb"],
                    "avg_cpu_pct":  hw["avg_cpu_pct"],
                    "rag_used":     use_rag,
                })
                set_status(cp, pid, vname, S_SUCCESS)
            except Exception as e:
                log(f"    Judge    : ✗ failed — {e}")
                set_status(cp, pid, vname, S_ERROR)
                prompt_ok = False

            time.sleep(2)

        # ── Prompt summary ────────────────────────────────────────────────────
        if prompt_ok and prompt_complete(cp, pid):
            done_count += 1
            log()
            log(f"  ✓ Prompt {pid} complete — all variants succeeded.")
        else:
            log()
            log(f"  ✗ Prompt {pid} incomplete — failed variants will retry on next run.")

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
        log("     Command:  taskpolicy -b python run_rq2.py")
    else:
        log("  ✓ All prompts complete!")
        log("     Next step:  python analysis_rq2.py")
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
        print("\n\nInterrupted. Checkpoint saved — re-run to resume.")
        sys.exit(0)