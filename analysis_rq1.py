"""
analysis_rq1.py — RQ1 result analysis after run_experiments.py completes.

Reads:
    results/rq1_scores.csv
    results/rq2_metrics.csv

Prints:
    - Average quality scores per model (overall + per category)
    - Average hardware metrics per model (latency, RAM, CPU)
    - Mann-Whitney U test: each local model vs GPT-OSS-120B

Run:
    python analysis_rq1.py
"""

import csv
from pathlib import Path

RESULTS = Path("results")

LOCAL_MODEL_NAMES = ["Qwen2.5-0.5B", "Llama-3.2-1B"]
CLOUD_MODEL_NAME  = "GPT-OSS-120B"


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values):
    return round(sum(values) / len(values), 3) if values else 0.0


def main():
    scores_path  = RESULTS / "rq1_scores.csv"
    metrics_path = RESULTS / "rq2_metrics.csv"

    if not scores_path.exists():
        print("ERROR: results/rq1_scores.csv not found. Run run_experiments.py first.")
        return

    scores  = load_csv(scores_path)
    metrics = load_csv(metrics_path) if metrics_path.exists() else []

    # ── Sanity check: show what model names are in the CSV ────────────────────
    found_models = sorted({r["model"] for r in scores})
    print(f"\n[INFO] Models found in rq1_scores.csv: {found_models}")

    models     = found_models
    categories = sorted({r["category"] for r in scores})

    # ── Quality scores ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RQ1 — QUALITY SCORES (judge: Groq Qwen3.8-27B, scale 1–5)")
    print("=" * 60)

    print(f"\n{'Model':<22} {'Accuracy':>9} {'Clarity':>9} {'Completeness':>13} {'Overall':>9}")
    print("-" * 65)
    for m in models:
        rows = [r for r in scores if r["model"] == m]
        acc  = mean([float(r["accuracy"])     for r in rows])
        cla  = mean([float(r["clarity"])      for r in rows])
        com  = mean([float(r["completeness"]) for r in rows])
        ov   = mean([float(r["overall"])      for r in rows])
        print(f"{m:<22} {acc:>9.3f} {cla:>9.3f} {com:>13.3f} {ov:>9.3f}")

    print("\n\nPer-category overall scores:")
    print(f"{'Category':<14}", end="")
    for m in models:
        print(f"  {m[:16]:>16}", end="")
    print()
    print("-" * (14 + 18 * len(models)))

    for cat in categories:
        print(f"{cat:<14}", end="")
        for m in models:
            rows = [r for r in scores if r["model"] == m and r["category"] == cat]
            ov   = mean([float(r["overall"]) for r in rows]) if rows else 0.0
            print(f"  {ov:>16.3f}", end="")
        print()

    # ── Hardware metrics ──────────────────────────────────────────────────────
    if metrics:
        print("\n\n" + "=" * 60)
        print("RQ1/RQ2 — HARDWARE METRICS (local models only)")
        print("=" * 60)
        local_models = sorted({r["model"] for r in metrics})
        print(f"\n{'Model':<22} {'Latency(s)':>11} {'Peak RAM(MB)':>13} {'Avg CPU(%)':>11}")
        print("-" * 60)
        for m in local_models:
            rows = [r for r in metrics if r["model"] == m]
            lat  = mean([float(r["latency_s"])   for r in rows])
            ram  = mean([float(r["peak_ram_mb"]) for r in rows])
            cpu  = mean([float(r["avg_cpu_pct"]) for r in rows])
            print(f"{m:<22} {lat:>11.3f} {ram:>13.1f} {cpu:>11.1f}")

    # ── Statistical test (Mann-Whitney U) ─────────────────────────────────────
    try:
        from scipy import stats

        print("\n\n" + "=" * 60)
        print("STATISTICAL SIGNIFICANCE — Mann-Whitney U test")
        print("(non-parametric; H0: no difference in overall scores)")
        print("=" * 60)

        cloud_scores = [float(r["overall"]) for r in scores
                        if r["model"] == CLOUD_MODEL_NAME]

        if not cloud_scores:
            print(f"\n  ERROR: No scores found for cloud model '{CLOUD_MODEL_NAME}'.")
            print(f"  Models in CSV: {found_models}")
        else:
            print(f"\n  Cloud model ({CLOUD_MODEL_NAME}): n={len(cloud_scores)} scores")
            for m in LOCAL_MODEL_NAMES:
                local_s = [float(r["overall"]) for r in scores if r["model"] == m]
                if not local_s:
                    print(f"\n  WARNING: No scores found for '{m}' — skipping.")
                    continue
                stat, p = stats.mannwhitneyu(local_s, cloud_scores,
                                             alternative="two-sided")
                sig = ("*** p<0.001" if p < 0.001
                       else "** p<0.01" if p < 0.01
                       else "* p<0.05"  if p < 0.05
                       else "n.s.")
                print(f"\n  {m} vs {CLOUD_MODEL_NAME}:")
                print(f"    n={len(local_s)} vs n={len(cloud_scores)}")
                print(f"    U={stat:.1f},  p={p:.4f}  {sig}")

    except ImportError:
        print("\n  (Install scipy for statistical tests: pip install scipy)")
    except Exception as e:
        print(f"\n  ERROR during statistical test: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
