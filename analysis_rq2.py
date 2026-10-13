"""
analysis_rq2.py — Statistical analysis for RQ2 results.

What this prints
────────────────
  1. Quality comparison table   — mean scores per variant vs RQ1 baseline
  2. Category breakdown         — per-category overall score, all variants
  3. Statistical tests          — Mann-Whitney U: each RQ2 variant vs baseline
  4. Hardware comparison table  — latency, RAM, CPU per variant vs baseline
  5. Delta summary              — Δ score and Δ RAM for each augmentation

Baseline: Llama-1B-FP16 from results/rq1_scores.csv
RQ2 data: results/rq2_scores.csv + results/rq2_metrics.csv

Usage:
    python analysis_rq2.py
"""

import csv
from pathlib import Path
from collections import defaultdict
from scipy.stats import mannwhitneyu

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")

RQ1_SCORES_PATH  = RESULTS_DIR / "rq1_scores.csv"
RQ2_SCORES_PATH  = RESULTS_DIR / "rq2_scores.csv"
RQ2_METRICS_PATH = RESULTS_DIR / "rq2_metrics.csv"

# RQ1 model name for the Llama FP16 baseline
BASELINE_NAME = "Llama-3.2-1B"

# RQ2 variant names (must match run_rq2.py)
VARIANT_NAMES = [
    "Llama-1B-Q4",
    "Llama-1B-FP16-RAG",
    "Llama-1B-Q4-RAG",
]

VARIANT_LABELS = {
    "Llama-1B-Q4":       "Variant A: Q4_0, no RAG",
    "Llama-1B-FP16-RAG": "Variant B: FP16 + RAG",
    "Llama-1B-Q4-RAG":   "Variant C: Q4_0 + RAG",
}

CATEGORIES = [
    "writing", "roleplay", "reasoning", "math",
    "coding", "extraction", "stem", "humanities",
]

SCORE_COLS = ["accuracy", "clarity", "completeness", "overall"]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}\nRun the experiment first.")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _floatify(rows: list[dict], cols: list[str]) -> list[dict]:
    """Cast specified columns to float in-place; skip rows that fail."""
    clean = []
    for row in rows:
        try:
            for c in cols:
                row[c] = float(row[c])
            clean.append(row)
        except (ValueError, KeyError):
            pass
    return clean


def load_data():
    rq1 = _load_csv(RQ1_SCORES_PATH)
    rq2s = _load_csv(RQ2_SCORES_PATH)
    rq2m = _load_csv(RQ2_METRICS_PATH)

    rq1  = _floatify(rq1,  SCORE_COLS)
    rq2s = _floatify(rq2s, SCORE_COLS)
    rq2m = _floatify(rq2m, ["latency_s", "peak_ram_mb", "avg_cpu_pct"])

    # Baseline: RQ1 rows for Llama FP16
    baseline = [r for r in rq1 if r["model"] == BASELINE_NAME]
    if not baseline:
        raise ValueError(
            f"No rows for '{BASELINE_NAME}' in rq1_scores.csv. "
            "Check the model name in the CSV header."
        )

    # RQ2 rows per variant
    variants = {name: [r for r in rq2s if r["model"] == name]
                for name in VARIANT_NAMES}

    # Metrics per variant
    metrics = {name: [r for r in rq2m if r["model"] == name]
               for name in VARIANT_NAMES}

    return baseline, variants, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def mean(xs) -> float:
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else float("nan")


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def mann_whitney(a: list, b: list):
    """Two-sided Mann-Whitney U. Returns (U, p)."""
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return round(stat, 1), round(p, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

_SEP  = "═" * 74
_DASH = "─" * 74


def _header(title: str):
    print()
    print(_SEP)
    print(f"  {title}")
    print(_SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Quality comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_quality_table(baseline, variants):
    _header("1. Overall Quality Scores  (LLM-as-Judge, 1–5 scale)")

    col_w = 10
    header = f"  {'Model':<26} {'Accuracy':>{col_w}} {'Clarity':>{col_w}} {'Completeness':>{col_w}} {'Overall':>{col_w}}"
    print(header)
    print(f"  {'-'*26} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")

    def _print_row(label, rows):
        acc = mean(r["accuracy"]     for r in rows)
        cla = mean(r["clarity"]      for r in rows)
        com = mean(r["completeness"] for r in rows)
        ovr = mean(r["overall"]      for r in rows)
        print(f"  {label:<26} {acc:>{col_w}.3f} {cla:>{col_w}.3f} {com:>{col_w}.3f} {ovr:>{col_w}.3f}")

    _print_row(f"RQ1 Baseline (FP16, no RAG)", baseline)
    print(f"  {'-'*26} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")
    for name in VARIANT_NAMES:
        rows = variants[name]
        if rows:
            _print_row(VARIANT_LABELS[name][:26], rows)
        else:
            print(f"  {VARIANT_LABELS[name][:26]:<26} {'(no data)':>{col_w}}")

    print()
    print("  Note: Higher = better. Judge: Groq Qwen3.8-27B (same as RQ1).")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Category breakdown
# ─────────────────────────────────────────────────────────────────────────────

def print_category_breakdown(baseline, variants):
    _header("2. Overall Score by Category")

    all_models = [("Baseline (FP16)", baseline)] + [
        (VARIANT_LABELS[n][:18], variants[n]) for n in VARIANT_NAMES
    ]

    # Column widths
    cat_w  = 12
    mod_w  = 8

    # Header row
    head = f"  {'Category':<{cat_w}}"
    for label, _ in all_models:
        head += f"  {label[:mod_w]:>{mod_w}}"
    print(head)
    print(f"  {'-'*cat_w}" + f"  {'-'*mod_w}" * len(all_models))

    for cat in CATEGORIES:
        row = f"  {cat.capitalize():<{cat_w}}"
        for _, rows in all_models:
            cat_rows = [r for r in rows if r["category"] == cat]
            val = mean(r["overall"] for r in cat_rows) if cat_rows else float("nan")
            row += f"  {val:>{mod_w}.3f}" if cat_rows else f"  {'N/A':>{mod_w}}"
        print(row)

    print()
    print("  Bold differences across variants reveal which categories benefit")
    print("  most from quantization or RAG augmentation.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Statistical significance
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(baseline, variants):
    _header("3. Statistical Significance  (Mann-Whitney U, two-sided)")

    base_overall = [r["overall"] for r in baseline]

    print(f"  {'Comparison':<40} {'U':>10} {'p-value':>10} {'Sig':>5} {'n':>5}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*5} {'-'*5}")

    for name in VARIANT_NAMES:
        rows = variants[name]
        if not rows:
            print(f"  {VARIANT_LABELS[name][:40]:<40} {'(no data)':>10}")
            continue
        var_overall = [r["overall"] for r in rows]
        U, p = mann_whitney(base_overall, var_overall)
        stars = sig_stars(p)
        label = f"Baseline vs {name}"
        print(f"  {label:<40} {U:>10.1f} {p:>10.4f} {stars:>5} {len(var_overall):>5}")

    print()
    print("  *** p < 0.001  ** p < 0.01  * p < 0.05  ns = not significant")
    print("  Mann-Whitney U used because 1–5 Likert scores violate normality")
    print("  (Shapiro-Wilk on RQ1 data confirms non-normal distributions).")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Hardware metrics
# ─────────────────────────────────────────────────────────────────────────────

def print_hardware_table(baseline_metrics_rq1, rq2_metrics):
    """
    baseline_metrics_rq1: rows from rq2_metrics.csv for Llama-3.2-1B from RQ1
                           (loaded separately from rq1_metrics.csv if available,
                            or we show 'see rq2_metrics.csv')
    rq2_metrics:          dict {variant_name: [rows]}
    """
    _header("4. Hardware Metrics  (Apple M2, 8 GB unified memory)")

    col_w = 12
    header = f"  {'Model':<26} {'Latency (s)':>{col_w}} {'Peak RAM (MB)':>{col_w}} {'Avg CPU (%)':>{col_w}}"
    print(header)
    print(f"  {'-'*26} {'-'*col_w} {'-'*col_w} {'-'*col_w}")

    def _hw_row(label, rows):
        if not rows:
            print(f"  {label:<26} {'(no data)':>{col_w}}")
            return
        lat = mean(float(r["latency_s"])   for r in rows)
        ram = mean(float(r["peak_ram_mb"]) for r in rows)
        cpu = mean(float(r["avg_cpu_pct"]) for r in rows)
        print(f"  {label:<26} {lat:>{col_w}.3f} {ram:>{col_w}.1f} {cpu:>{col_w}.1f}")

    if baseline_metrics_rq1:
        _hw_row("RQ1 Baseline (FP16, no RAG)", baseline_metrics_rq1)
    else:
        print(f"  {'RQ1 Baseline (FP16, no RAG)':<26} {'see rq2_metrics.csv':>{col_w}}")

    print(f"  {'-'*26} {'-'*col_w} {'-'*col_w} {'-'*col_w}")
    for name in VARIANT_NAMES:
        _hw_row(VARIANT_LABELS[name][:26], rq2_metrics[name])

    print()
    print("  Latency: wall-clock seconds per inference  (lower = better)")
    print("  Peak RAM: max Ollama process RSS in MB     (lower = better)")
    print("  Avg CPU: mean CPU% during inference")
    print()
    print("  Expected: Q4_0 variants use ~700 MB vs ~2545 MB for FP16.")
    print("  RAG adds embedding retrieval time (~50–200 ms, negligible vs inference).")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Delta summary
# ─────────────────────────────────────────────────────────────────────────────

def print_delta_summary(baseline, variants, rq1_metrics, rq2_metrics):
    _header("5. Augmentation Delta Summary  (vs FP16 baseline, no RAG)")

    base_q   = mean(r["overall"] for r in baseline)
    base_ram = mean(float(r["peak_ram_mb"]) for r in rq1_metrics) if rq1_metrics else None
    base_lat = mean(float(r["latency_s"])   for r in rq1_metrics) if rq1_metrics else None

    print(f"  {'Variant':<26} {'Δ Overall':>10} {'Δ RAM (MB)':>12} {'Δ Latency(s)':>14} {'Interpretation'}")
    print(f"  {'-'*26} {'-'*10} {'-'*12} {'-'*14} {'-'*24}")

    for name in VARIANT_NAMES:
        srows = variants[name]
        mrows = rq2_metrics[name]

        if not srows:
            print(f"  {VARIANT_LABELS[name][:26]:<26} {'(no data)':>10}")
            continue

        var_q   = mean(r["overall"] for r in srows)
        dq      = round(var_q - base_q, 3)
        dq_str  = f"{dq:+.3f}"

        if mrows and base_ram is not None:
            var_ram  = mean(float(r["peak_ram_mb"]) for r in mrows)
            dram_str = f"{var_ram - base_ram:+.1f}"
            var_lat  = mean(float(r["latency_s"])   for r in mrows)
            dlat_str = f"{var_lat - base_lat:+.1f}"
        else:
            dram_str = "N/A"
            dlat_str = "N/A"

        # Interpretation
        if dq > 0.1:
            interp = "↑ quality gain"
        elif dq < -0.1:
            interp = "↓ quality drop"
        else:
            interp = "≈ no change"

        print(f"  {VARIANT_LABELS[name][:26]:<26} {dq_str:>10} {dram_str:>12} {dlat_str:>14}   {interp}")

    print()
    print("  Positive Δ Overall = improvement over baseline.")
    print("  Negative Δ RAM     = memory freed (Q4_0 saves ~1800 MB vs FP16).")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(_SEP)
    print("  RQ2 Analysis — Augmenting Llama-3.2-1B on Low-Spec Hardware")
    print(_SEP)

    baseline, variants, rq2_metrics = load_data()

    # Try to load RQ1 hardware metrics for Llama baseline (may not exist)
    rq1_metrics_path = RESULTS_DIR / "rq2_metrics.csv"   # RQ1 hardware was in rq2_metrics
    rq1_hw = []
    try:
        all_hw = _load_csv(rq1_metrics_path)
        rq1_hw = _floatify(
            [r for r in all_hw if r["model"] == BASELINE_NAME],
            ["latency_s", "peak_ram_mb", "avg_cpu_pct"]
        )
    except FileNotFoundError:
        pass

    n_baseline = len(baseline)
    n_variants = {name: len(rows) for name, rows in variants.items()}

    print(f"\n  Baseline rows loaded  : {n_baseline}  ({BASELINE_NAME} from rq1_scores.csv)")
    for name, count in n_variants.items():
        print(f"  {name:<22} : {count} rows")

    if not all(n > 0 for n in n_variants.values()):
        missing = [n for n, c in n_variants.items() if c == 0]
        print(f"\n  ⚠  No data yet for: {', '.join(missing)}")
        print("     Run:  taskpolicy -b python run_rq2.py")
        if n_baseline > 0 and any(n > 0 for n in n_variants.values()):
            print("     (Showing partial results for completed variants)\n")
        else:
            return

    print_quality_table(baseline, variants)
    print_category_breakdown(baseline, variants)
    print_stats(baseline, variants)
    print_hardware_table(rq1_hw, rq2_metrics)
    print_delta_summary(baseline, variants, rq1_hw, rq2_metrics)

    print()
    print(_SEP)
    print("  Analysis complete.")
    print(_SEP)


if __name__ == "__main__":
    main()