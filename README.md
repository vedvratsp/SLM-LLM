# Comparing & Augmenting LLM Capabilities on Low-Spec Hardware

**MSc Artificial Intelligence — TU Dublin, 2026**  
Vedvrat Singh Parihar · A00051088  
Supervisor review repository

---

## Overview

This repository contains all experimental code for a thesis investigating whether small language models (SLMs) running entirely on consumer hardware — an Apple M2 MacBook Pro with 8 GB unified memory — can be made practically useful through augmentation techniques (4-bit quantisation and Retrieval-Augmented Generation), and how they compare to a large cloud model on quality, latency, and trustworthiness.

The thesis is structured around three research questions:

| RQ | Question | Technique |
|---|---|---|
| **RQ1** | How do two local SLMs compare to a large cloud model on response quality and hardware cost? | LLM-as-Judge evaluation |
| **RQ2** | Can 4-bit quantisation and RAG improve the best local SLM's quality-efficiency trade-off? | Quantisation + RAG augmentation |
| **RQ3** | Which model do blind human participants trust more, and does it pass an EU ALTAI ethics check? | Blind A/B user study + ALTAI framework |

---

## Models

### Local (via [Ollama](https://ollama.com/), Apple Silicon, FP16, unquantised for RQ1)
| Model | Ollama tag | Size (approx.) |
|---|---|---|
| Qwen2.5-0.5B | `qwen2.5:0.5b-instruct-fp16` | ~1 GB |
| Llama-3.2-1B | `llama3.2:1b-instruct-fp16` | ~2 GB |

### Cloud (RQ1 baseline & RQ3 comparison)
| Model | Provider |
|---|---|
| GPT-OSS-120B | Groq API (`openai/gpt-oss-120b`) |

### Judge (LLM-as-Judge, independent of tested models)
| Model | Provider | Rationale |
|---|---|---|
| Qwen3.8-27B | Groq API | Validated by Zheng et al. (MT-Bench, NeurIPS 2023) LLM-as-judge methodology |

---

## Evaluation Benchmark

All 80 MT-Bench Turn-1 questions are used (10 per category). Multi-turn (Turn 2) evaluation is excluded because Turn 2 responses are conditioned on Turn 1 quality, which confounds cross-model comparison.

**Categories:** Writing · Roleplay · Reasoning · Math · Coding · Extraction · STEM · Humanities

> **Citation:** Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS 2023.  
> https://arxiv.org/abs/2306.05685

---

## RQ2 Augmentation Variants

The Llama-3.2-1B FP16 result from RQ1 serves as the baseline. Three augmented variants are compared against it:

| Variant | Label | Quantisation | RAG |
|---|---|---|---|
| A | Llama-1B-Q4 | Q4\_K\_M | ✗ |
| B | Llama-1B-FP16-RAG | FP16 | ✓ |
| C | Llama-1B-Q4-RAG | Q4\_K\_M | ✓ |

**RAG corpus:** 80 Wikipedia articles (10 per MT-Bench category), chunked to ~300 words, embedded with `all-MiniLM-L6-v2` (Reimers & Gurevych, EMNLP 2019), and stored in ChromaDB with cosine similarity. Top-3 chunks are retrieved per prompt. The judge scores the *original* question — not the RAG-augmented prompt — so scores remain comparable across variants.

> **Citation:** Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020. https://arxiv.org/abs/2005.11401  
> **Citation:** Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023. https://arxiv.org/abs/2305.14314

---

## RQ3 — Trustworthiness Study

Six ethics-approved custom prompts are run through all models. Responses are de-identified and presented as "Response A / Response B" in a blind user study. Participants rate each response on five Likert items grounded in Mayer et al.'s (1995) Ability–Benevolence–Integrity trust model. An EU ALTAI checklist (Transparency, Privacy, Fairness, Robustness) provides a qualitative ethics supplement.

> **Citation:** Mayer et al., *An Integrative Model of Organizational Trust*, AMR 1995.  
> **Citation:** Siau & Wang, *Building Trust in Artificial Intelligence, Machine Learning, and Robotics*, CACM 2018.

---

## Repository Structure

```
.
├── run_experiments.py      # RQ1: local + cloud inference, LLM-as-judge, hardware metrics
├── analysis_rq1.py         # RQ1: summary tables and Mann-Whitney U tests
│
├── build_rag_corpus.py     # RQ2 setup: download Wikipedia articles, build ChromaDB
├── rag_engine.py           # RQ2: retrieval function (top-k cosine similarity)
├── run_rq2.py              # RQ2: augmented inference (Q4 / RAG variants)
├── analysis_rq2.py         # RQ2: delta tables and statistical significance
│
├── rq3_prompts.py          # RQ3: 6 ethics-approved custom prompts
├── rq3_run.py              # RQ3: inference + judge on custom prompts
├── rq3_prep.py             # RQ3: build blind A/B survey from responses
│
├── mt_bench_prompts.py     # All 80 MT-Bench Turn-1 questions (IDs 81–160)
├── hardware_monitor.py     # psutil-based latency / RAM / CPU measurement
├── recollect_metrics.py    # Utility: re-measure hardware metrics from saved responses
│
├── requirements.txt
└── .gitignore
```

> **Note:** `corpus/` and `rag_db/` are excluded from version control (see `.gitignore`). Regenerate them with `python build_rag_corpus.py` before running RQ2 scripts.

---

## Setup

### Prerequisites
- macOS (Apple Silicon recommended) with [Ollama](https://ollama.com/) installed
- Python 3.11+
- A Groq API key (free tier sufficient): https://console.groq.com

### Install dependencies

```bash
pip install -r requirements.txt
```

### Pull local models

```bash
ollama pull qwen2.5:0.5b-instruct-fp16
ollama pull llama3.2:1b-instruct-fp16

# RQ2 quantised variants
ollama pull llama3.2:1b          # Q4_K_M
```

### Environment variables

Create a `.env` file in the project root (never commit this):

```
GROQ_API_KEY=your_groq_key_here
```

---

## Running the Experiments

### RQ1 — Baseline comparison

```bash
taskpolicy -b python run_experiments.py
```

Outputs to `results/rq1_responses.csv`, `results/rq1_scores.csv`, `results/rq2_metrics.csv`.  
Re-run any number of times — the checkpoint system skips already-completed (prompt, model) pairs.

```bash
python analysis_rq1.py
```

### RQ2 — Augmentation ablation

```bash
# One-time corpus setup (safe to re-run after a crash)
python build_rag_corpus.py

# Run augmented variants
taskpolicy -b python run_rq2.py

python analysis_rq2.py
```

### RQ3 — Trustworthiness study

```bash
taskpolicy -b python rq3_run.py
python rq3_prep.py    # generates the blind A/B survey file
```

---

## Statistical Approach

Response quality scores (1–5 Likert scale) are non-normally distributed (confirmed by Shapiro-Wilk). All pairwise significance tests use the **Mann-Whitney U test** (two-sided). The threshold for significance is *p* < 0.05.

---

## Hardware

All local inference was conducted on an Apple M2 MacBook Pro with 8 GB unified memory — a deliberately constrained, consumer-grade device representative of privacy-first or resource-limited deployment scenarios. No dedicated GPU or cloud compute was used for local model inference.

| Metric | Measurement method |
|---|---|
| Latency | Wall-clock seconds (prompt send → response end) |
| Peak RAM | Maximum Ollama process RSS via `psutil` |
| Avg CPU | Mean CPU % sampled at 0.5 s intervals during inference |

---

## Privacy & Offline Capability

Local models run entirely on-device. No user query data leaves the machine. This property is one of the core arguments for small, locally deployed models — particularly relevant for healthcare, legal, and air-gapped environments.

---

## License

This repository is shared for academic review purposes. All experimental code is original work by the author. MT-Bench prompts are used under the original FastChat/LMSYS licence for academic research.