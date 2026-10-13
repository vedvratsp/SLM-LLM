"""
build_rag_corpus.py — One-time setup for RQ2 RAG knowledge base.

Phase 1: Downloads 80 Wikipedia articles to ./corpus/<category>/<title>.txt
          (skips files already on disk — safe to re-run after a crash).
Phase 2: Reads local files, chunks, embeds, stores in ChromaDB at ./rag_db/.

Run once before run_rq2.py:
    python build_rag_corpus.py

Citations
---------
Reimers & Gurevych, "Sentence-BERT", EMNLP 2019, https://arxiv.org/abs/1908.10084
"""

import re
import time
from pathlib import Path

import requests

CATEGORY_ARTICLES = {
    "writing": [
        "Writing", "Essay", "Rhetoric", "Narrative", "Technical writing",
        "Creative writing", "Argumentation theory", "Style guide",
        "Academic writing", "Expository writing",
    ],
    "roleplay": [
        "Role-playing", "Character (arts)", "Improvisation", "Dialogue",
        "Persona", "Narrative", "Social simulation", "Interactive fiction",
        "Game master", "Cosplay",
    ],
    "reasoning": [
        "Logic", "Deductive reasoning", "Inductive reasoning",
        "Abductive reasoning", "Critical thinking", "Cognitive bias",
        "Fallacy", "Syllogism", "Problem solving", "Decision-making",
    ],
    "math": [
        "Mathematics", "Arithmetic", "Algebra", "Calculus", "Probability",
        "Statistics", "Number theory", "Geometry", "Linear algebra",
        "Discrete mathematics",
    ],
    "coding": [
        "Computer programming", "Algorithm", "Data structure",
        "Time complexity", "Sorting algorithm", "Recursion (computer science)",
        "Object-oriented programming", "Functional programming",
        "Software testing", "Debugging",
    ],
    "extraction": [
        "Information extraction", "Named-entity recognition", "Text mining",
        "Natural language processing", "Parsing", "Regular expression",
        "Data extraction", "Knowledge extraction", "Coreference resolution",
        "Relation extraction",
    ],
    "stem": [
        "Science", "Physics", "Chemistry", "Biology", "Engineering",
        "Thermodynamics", "Electromagnetism", "Evolution", "Genetics",
        "Quantum mechanics",
    ],
    "humanities": [
        "History", "Philosophy", "Literature", "Ethics", "Sociology",
        "Political science", "Economics", "Anthropology", "Linguistics",
        "Art history",
    ],
}

CHUNK_SIZE_WORDS = 300
OVERLAP_WORDS    = 10
CORPUS_DIR       = Path("corpus")
RAG_DB_DIR       = Path("rag_db")

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS  = {"User-Agent": "thesis-rag/1.0 (academic research)"}
SLEEP_S  = 3.0   # seconds between requests — well under Wikipedia's rate limit


# ── Phase 1: fetch from Wikipedia and save to disk ────────────────────────────

def _safe_filename(title: str) -> str:
    return re.sub(r"[^a-z0-9\-]", "_", title.lower()) + ".txt"


def _fetch_one(title: str) -> str:
    """Fetch full plain-text for one Wikipedia article. Returns '' on failure."""
    params = {
        "action":        "query",
        "prop":          "extracts",
        "explaintext":   "1",
        "redirects":     "1",
        "titles":        title,
        "format":        "json",
        "formatversion": "2",
    }
    for attempt in range(1, 5):
        try:
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", [])
            if not pages or pages[0].get("missing"):
                return ""
            return pages[0].get("extract", "").strip()
        except Exception as e:
            if attempt == 4:
                return ""
            wait = 2 ** attempt
            print(f"(retry {attempt}/3 in {wait}s…)", end=" ", flush=True)
            time.sleep(wait)
    return ""


def download_corpus():
    print("\n  ── Phase 1: Downloading Wikipedia articles ──")
    ok = fail = 0
    for category, titles in CATEGORY_ARTICLES.items():
        cat_dir = CORPUS_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for title in titles:
            fpath = cat_dir / _safe_filename(title)
            if fpath.exists() and fpath.stat().st_size > 200:
                print(f"    ✓ [cached] {title}")
                ok += 1
                continue
            print(f"    Fetching: {title}…", end=" ", flush=True)
            text = _fetch_one(title)
            if text:
                fpath.write_text(text, encoding="utf-8")
                print(f"✓  ({len(text.split())} words)")
                ok += 1
            else:
                print("✗ (failed)")
                fail += 1
            time.sleep(SLEEP_S)
    print(f"\n  Phase 1 done: {ok} ok, {fail} failed")
    if fail:
        print("  Re-run the script to retry failed articles.")


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    text = re.sub(r"={2,}[^=]+={2,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start : start + CHUNK_SIZE_WORDS])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start += CHUNK_SIZE_WORDS - OVERLAP_WORDS
    return chunks


# ── Phase 2: embed and store in ChromaDB ─────────────────────────────────────

def build_chromadb():
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    print("\n  ── Phase 2: Building ChromaDB ──")
    print("  Loading embedding model…")
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2", device="cpu"
    )
    print("  ✓ Model ready")

    RAG_DB_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(RAG_DB_DIR))
    total = 0

    for category, titles in CATEGORY_ARTICLES.items():
        coll_name = f"rq2_{category}"
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass
        coll = client.create_collection(
            name=coll_name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        cat_n = 0
        print(f"\n  ── {category.upper()} ──")
        for title in titles:
            fpath = CORPUS_DIR / category / _safe_filename(title)
            if not fpath.exists():
                print(f"    ✗ {title} (missing — re-run Phase 1)")
                continue
            chunks = _chunk_text(fpath.read_text(encoding="utf-8"))
            if not chunks:
                print(f"    ✗ {title} (no chunks)")
                continue
            safe = re.sub(r"[^a-z0-9_]", "_", title.lower())
            coll.add(
                documents=chunks,
                ids=[f"{category}_{safe}_{i}" for i in range(len(chunks))],
                metadatas=[{"article": title, "category": category, "chunk_index": i}
                           for i in range(len(chunks))],
            )
            cat_n += len(chunks)
            print(f"    ✓  {title}: {len(chunks)} chunks")
        print(f"    → {coll_name}: {cat_n} chunks")
        total += cat_n

    print(f"\n  Phase 2 done: {total} chunks in {RAG_DB_DIR}/")
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Building RAG corpus for RQ2")
    print("=" * 60)
    download_corpus()
    total = build_chromadb()
    print()
    print("=" * 60)
    print(f"  Done. {total} chunks stored.")
    print("  Verify with:")
    print('  python -c "import chromadb; c=chromadb.PersistentClient(\'rag_db\'); [print(x.name, x.count()) for x in c.list_collections()]"')
    print("  Next: python run_rq2.py")
    print("=" * 60)


if __name__ == "__main__":
    main()