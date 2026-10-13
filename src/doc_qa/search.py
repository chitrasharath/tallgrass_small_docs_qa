"""Search passages with BM25 keywords or embeddings (local model or the LiteLLM proxy)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from doc_qa.config import llm_settings
from doc_qa.embedders import ApiEmbedder, Embedder, LocalEmbedder
from doc_qa.models import Passage

DEFAULT_PASSAGES = Path("data/processed/passages.jsonl")

_TOKEN = re.compile(r"[a-z0-9$]+(?:[.\-:'][a-z0-9]+)*")
_STOPWORDS = frozenset(
    """a an and are as at be been but by can do does for from has have how i if in into is it
    its me my of on or our so that the their them then there these this to was we were what
    when where which who will with you your""".split()
)


def load_passages(path: Path = DEFAULT_PASSAGES) -> list[Passage]:
    with path.open(encoding="utf-8") as handle:
        return [Passage(**json.loads(line)) for line in handle if line.strip()]


def index_text(passage: Passage) -> str:
    """What gets searched: the section heading in front of the passage text."""
    return f"{passage.section}\n{passage.text}" if passage.section else passage.text


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords; numbers like 38.50 and 555-0142 stay whole."""
    tokens = []
    for token in _TOKEN.findall(text.lower()):
        token = token.strip("$")
        if not token or token in _STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


@dataclass
class SearchHit:
    rank: int
    score: float
    passage: Passage


class Searcher(Protocol):
    name: str

    def search(self, query: str, k: int = 3) -> list[SearchHit]: ...


def _top_k(passages: list[Passage], scores: np.ndarray, k: int) -> list[SearchHit]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [SearchHit(rank=i + 1, score=float(scores[j]), passage=passages[j]) for i, j in enumerate(order)]


class BM25Search:
    name = "bm25"

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.term_counts = [Counter(tokenize(index_text(p))) for p in passages]
        self.lengths = np.array([sum(c.values()) for c in self.term_counts], dtype=float)
        self.average_length = float(self.lengths.mean()) if passages else 0.0
        document_frequency = Counter(term for counts in self.term_counts for term in counts)
        n = len(passages)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.passages))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            frequency = np.array([counts.get(term, 0) for counts in self.term_counts], dtype=float)
            norm = self.k1 * (1 - self.b + self.b * self.lengths / self.average_length)
            scores += idf * frequency * (self.k1 + 1) / (frequency + norm)
        return scores

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        return _top_k(self.passages, self.scores(query), k)


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]+", "_", model_name.split("/")[-1])


def _normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)


class EmbeddingSearch:
    """Cosine similarity between the query and passage embeddings.

    Passage and query embeddings are cached on disk per model, so a remote model is
    only called for text it has not embedded before.
    """

    def __init__(
        self,
        passages: list[Passage],
        embedder: Embedder,
        name: str,
        cache_dir: Path | None = DEFAULT_PASSAGES.parent,
    ):
        self.passages = passages
        self.embedder = embedder
        self.name = name
        slug = model_slug(embedder.model_name)
        texts = [index_text(p) for p in passages]
        key = hashlib.sha256(json.dumps([embedder.model_name, texts]).encode()).hexdigest()[:16]
        passage_cache = cache_dir / f"embeddings_{slug}_{key}.npy" if cache_dir else None
        self._query_cache_path = cache_dir / f"query_embeddings_{slug}.json" if cache_dir else None
        self._query_vectors: dict[str, list[float]] = (
            json.loads(self._query_cache_path.read_text())
            if self._query_cache_path and self._query_cache_path.exists()
            else {}
        )

        if passage_cache and passage_cache.exists():
            self.matrix = np.load(passage_cache)
        else:
            self.matrix = _normalize(embedder.embed_passages(texts))
            if passage_cache:
                cache_dir.mkdir(parents=True, exist_ok=True)
                for stale in cache_dir.glob(f"embeddings_{slug}_*.npy"):
                    stale.unlink()
                np.save(passage_cache, self.matrix)

    def prefetch(self, queries: list[str]) -> None:
        """Embed all uncached queries in one batch instead of one request per query."""
        missing = list(dict.fromkeys(q for q in queries if q not in self._query_vectors))
        if not missing:
            return
        for query, vector in zip(missing, self.embedder.embed_queries(missing)):
            self._query_vectors[query] = [float(x) for x in vector]
        if self._query_cache_path:
            self._query_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._query_cache_path.write_text(json.dumps(self._query_vectors))

    def scores(self, query: str) -> np.ndarray:
        self.prefetch([query])
        return self.matrix @ _normalize(np.array(self._query_vectors[query]))

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        return _top_k(self.passages, self.scores(query), k)


SEARCH_METHODS = ("bm25", "local-embedding", "api-embedding")


def make_searcher(method: str, passages: list[Passage]) -> Searcher:
    if method == "bm25":
        return BM25Search(passages)
    if method == "local-embedding":
        return EmbeddingSearch(passages, LocalEmbedder(), name=method)
    if method == "api-embedding":
        return EmbeddingSearch(passages, ApiEmbedder(), name=method)
    raise ValueError(f"Unknown search method: {method}")


def resolve_methods(choice: str) -> list[str]:
    """'all' means every method that can run: api-embedding needs LLM_API_KEY."""
    if choice != "all":
        return [choice]
    if llm_settings().api_key:
        return list(SEARCH_METHODS)
    print("LLM_API_KEY is not set, skipping api-embedding (see .example.env).")
    return [m for m in SEARCH_METHODS if m != "api-embedding"]
