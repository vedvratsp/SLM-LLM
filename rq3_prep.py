"""
rq3_prep.py — Prepare randomised A/B response pairs for the RQ3 survey.

Reads:  results/rq1_responses.csv   (produced by run_experiments.py)
        results/rq1_scores.csv      (judge scores)
Output: results/rq3_survey_responses.csv   ← paste these into Google Form
        results/rq3_key.csv                ← ground-truth A/B mapping (keep private)

The 6 custom prompts (ethics-approved) live in rq3_prompts.py.
This script selects responses for those 6 prompt IDs from the full rq1 CSV,
which contains responses for all 80 MT-Bench prompts.

For each of the 6 RQ3 prompts:
  - Picks ONE local model response (highest overall judge score;
    if tied, prefers Phi-3-mini as the larger model).
  - Randomly assigns it to slot A or slot B.
  - Gemini goes in the other slot.

The key file records which slot was local and which was cloud, so you can
decode the survey results after data collection.

IMPORTANT: rq3_prompts.py IDs map to mt_bench_prompts.py IDs as follows
(the 6 custom prompts are stored separately; their prompt_ids in rq1_responses.csv
come from their position in mt_bench_prompts.py if you wish to cross-reference,
OR you run rq3_run.py to generate responses for the 6 custom prompts specifically).

Usage: python rq3_prep.py
"""

import csv
import random
from pathlib import Path

from rq3_prompts import RQ3_PROMPTS

RESULTS = Path("results")
RESPONSES_FILE = RESULTS / "rq3_responses.csv"   # separate run for 6 custom prompts
SCORES_FILE    = RESULTS / "rq3_scores.csv"
SURVEY_FILE    = RESULTS / "rq3_survey_responses.csv"
KEY_FILE       = RESULTS / "rq3_key.csv"

# Preference order when scores are equal
MODEL_PREFERENCE = ["Phi-3-mini 3.8B", "TinyLlama 1.1B"]

random.seed(42)   # reproducible randomisation — record this seed in your thesis


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_local_response(prompt_id, responses, scores):
    """Return the response row for the best-scoring local model for this prompt."""
    local_responses = [r for r in responses
                       if r["prompt_id"] == str(prompt_id) and r["source"] == "local"]
    if not local_responses:
        raise ValueError(f"No local responses found for prompt {prompt_id}")

    local_scores = {s["model"]: float(s["overall"])
                    for s in scores
                    if s["prompt_id"] == str(prompt_id)}

    # Sort by score desc, then by MODEL_PREFERENCE for ties
    def sort_key(r):
        score = local_scores.get(r["model"], 0.0)
        pref  = MODEL_PREFERENCE.index(r["model"]) if r["model"] in MODEL_PREFERENCE else 99
        return (-score, pref)

    local_responses.sort(key=sort_key)
    return local_responses[0]


def main():
    if not RESPONSES_FILE.exists():
        print(f"ERROR: {RESPONSES_FILE} not found.")
        print("Run  python rq3_run.py  first to generate responses for the 6 custom prompts.")
        return

    responses = load_csv(RESPONSES_FILE)
    scores    = load_csv(SCORES_FILE) if SCORES_FILE.exists() else []

    survey_rows = []
    key_rows    = []

    for prompt in RQ3_PROMPTS:
        pid = prompt["id"]

        try:
            local_row = best_local_response(pid, responses, scores)
        except ValueError as e:
            print(f"WARNING: {e} — skipping.")
            continue

        gemini_rows = [r for r in responses
                       if r["prompt_id"] == str(pid) and r["source"] == "cloud"]
        if not gemini_rows:
            print(f"WARNING: No Gemini response for prompt {pid} — skipping.")
            continue
        gemini_row = gemini_rows[0]

        # Randomise slot assignment
        if random.random() < 0.5:
            slot_a, slot_b = local_row, gemini_row
            a_source, b_source = "local", "cloud"
            a_model,  b_model  = local_row["model"], "Gemini 3.8 Flash"
        else:
            slot_a, slot_b = gemini_row, local_row
            a_source, b_source = "cloud", "local"
            a_model,  b_model  = "Gemini 3.8 Flash", local_row["model"]

        survey_rows.append({
            "prompt_id":   pid,
            "category":    local_row["category"],
            "prompt_text": prompt["text"],
            "response_a":  slot_a["response"],
            "response_b":  slot_b["response"],
        })

        key_rows.append({
            "prompt_id":        pid,
            "category":         local_row["category"],
            "slot_a":           a_source,
            "slot_b":           b_source,
            "model_a":          a_model,
            "model_b":          b_model,
            "local_model_used": local_row["model"],
        })

    # Save
    _write_csv(SURVEY_FILE, survey_rows,
               ["prompt_id", "category", "prompt_text", "response_a", "response_b"])
    _write_csv(KEY_FILE, key_rows,
               ["prompt_id", "category", "slot_a", "slot_b", "model_a", "model_b", "local_model_used"])

    print(f"\nDone. {len(survey_rows)}/6 prompts processed.")
    print(f"  {SURVEY_FILE}  ← copy responses A/B into Google Form")
    print(f"  {KEY_FILE}     ← ground-truth mapping — keep private until analysis")
    print(f"\nRandom seed used: 42  (record this in your thesis methodology section)")


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
