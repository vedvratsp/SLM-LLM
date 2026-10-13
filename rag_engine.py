"""
rag_engine.py — Retrieval-Augmented Generation engine for RQ2.

API
───
    from rag_engine import RAGEngine

    engine = RAGEngine()
    augmented = engine.augment(category="coding", question="...")
    # augmented = "Context:\n<chunks>\n\nQuestion:\n<original question>"

Design
──────
• ChromaDB cosine-similarity retrieval — top-3 chunks per query.
  Three chunks ≈ 900 words of context; tested to stay within Llama's
  2048-token default context on Ollama without truncation.
• Lazy-loads the embedding model on first call so import is free.
• Falls back to the raw question if the collection for that category
  is not found (safety net; should not happen after build_rag_corpus.py).

Why top-3?
  Dense retrieval studies (Karpukhin et al., DPR 2020) show that precision
  plateaus after ~5 passages for open-domain QA, and more context hurts
  small models due to context-window pressure.  3 is a common default.

Citation
────────
Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question
Answering", EMNLP 2020, https://arxiv.org/abs/2004.04906
"""

from pathlib import Path

RAG_DB_DIR = Path("rag_db")
N_RESULTS   = 3   # top-k chunks to retrieve

# Prompt template — context prepended before the question.
# Keeps the question as the last line so the model sees it most recently
# in its attention window.
_AUGMENT_TEMPLATE = """\
Use the following reference information to help answer the question.
If the context is not directly relevant, rely on your own knowledge.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question:
{question}"""


class RAGEngine:
    """
    Retrieves relevant chunks from the per-category ChromaDB collection
    and returns an augmented prompt string ready to send to Ollama.
    """

    def __init__(self, db_path: str | Path = RAG_DB_DIR, n_results: int = N_RESULTS):
        self._db_path  = Path(db_path)
        self._n        = n_results
        self._client   = None
        self._embed_fn = None
        self._colls    = {}   # cache: category → chromadb.Collection

    # ── Lazy init ────────────────────────────────────────────────────────────

    def _init(self):
        """Load ChromaDB client and embedding model on first use."""
        if self._client is not None:
            return

        import chromadb
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        if not self._db_path.exists():
            raise FileNotFoundError(
                f"RAG database not found at '{self._db_path}'. "
                "Run  python build_rag_corpus.py  first."
            )

        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )
        self._client = chromadb.PersistentClient(path=str(self._db_path))

    def _get_collection(self, category: str):
        """Return the ChromaDB collection for a category (cached)."""
        if category in self._colls:
            return self._colls[category]

        self._init()
        try:
            coll = self._client.get_collection(
                name=f"rq2_{category}",
                embedding_function=self._embed_fn,
            )
            self._colls[category] = coll
            return coll
        except Exception:
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, category: str, question: str, n_results: int | None = None) -> list[str]:
        """
        Return the top-n most relevant text chunks for *question*
        from the *category* collection.

        Returns [] if the collection is not found.
        """
        n = n_results or self._n
        coll = self._get_collection(category)
        if coll is None:
            return []

        try:
            results = coll.query(
                query_texts=[question],
                n_results=min(n, coll.count()),  # guard: never ask more than available
            )
            return results["documents"][0]       # list of chunk strings
        except Exception:
            return []

    def augment(self, category: str, question: str) -> str:
        """
        Build and return the RAG-augmented prompt string.
        If no chunks are found, returns the bare question unchanged.
        """
        chunks = self.retrieve(category, question)
        if not chunks:
            return question

        context = "\n\n".join(chunks)
        return _AUGMENT_TEMPLATE.format(context=context, question=question)