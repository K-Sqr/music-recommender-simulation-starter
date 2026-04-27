"""Local RAG knowledge base for the agent.

Loads ``data/knowledge_base.json`` and provides a small TF-IDF-style
retriever. The agent calls :func:`retrieve` before deciding on weights so
its plan is grounded in concrete musical context (genre tendencies, decade
notes, mood/context advice) instead of pure keyword guessing.

The retrieved snippets are passed into the agent's planning prompt verbatim
so they visibly influence the chosen weights and the final explanations.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_TOKEN_RE = re.compile(r"[a-z0-9']+")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "is", "are", "be", "this", "that", "it", "its", "as", "at",
    "by", "from", "i", "want", "something", "music", "song", "songs",
    "track", "tracks", "playlist", "play", "give", "me", "some", "please",
    "really", "very", "kind", "type",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, regex-split, and filter stopwords/short tokens."""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


@dataclass
class KBDocument:
    """One entry from ``knowledge_base.json``."""

    id: str
    type: str
    name: str
    text: str

    def searchable(self) -> str:
        """Concatenated text used for token matching."""
        return f"{self.name} {self.type} {self.text}"


class KnowledgeBase:
    """Tiny in-memory retriever with TF-IDF scoring.

    The corpus is small (~30 docs) so we recompute IDF eagerly on load and
    score linearly per query. Retrieval is deterministic, which makes the
    eval harness reproducible without an embedding API.
    """

    def __init__(self, documents: List[KBDocument]):
        self.documents = documents
        self._tokenized: List[List[str]] = [
            _tokenize(d.searchable()) for d in documents
        ]
        self._df: Counter = Counter()
        for tokens in self._tokenized:
            for tok in set(tokens):
                self._df[tok] += 1
        self._n = max(len(documents), 1)

    @classmethod
    def from_json(cls, path: str | Path) -> "KnowledgeBase":
        """Load a knowledge base from a JSON file matching the bundled schema."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        docs = [
            KBDocument(
                id=d["id"],
                type=d.get("type", ""),
                name=d.get("name", ""),
                text=d.get("text", ""),
            )
            for d in payload.get("documents", [])
        ]
        return cls(docs)

    def _idf(self, token: str) -> float:
        df = self._df.get(token, 0)
        # Smooth IDF; +1 keeps unseen tokens at 0 contribution
        return math.log((self._n + 1) / (df + 1)) + 1.0

    def _score_doc(self, query_tokens: Iterable[str], doc_tokens: List[str]) -> float:
        if not doc_tokens:
            return 0.0
        doc_counts = Counter(doc_tokens)
        score = 0.0
        for tok in query_tokens:
            tf = doc_counts.get(tok, 0)
            if tf:
                score += (1 + math.log(tf)) * self._idf(tok)
        return score

    def retrieve(
        self,
        query: str,
        k: int = 3,
        types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Return the top-k matching documents for ``query``.

        Each result is shaped as
        ``{"id", "type", "name", "text", "score"}`` so it serialises cleanly
        into the agent's prompt. Documents with zero score are filtered out
        to avoid feeding the LLM noise.
        """
        if not query or not query.strip():
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scored: List[tuple[float, KBDocument]] = []
        for doc, tokens in zip(self.documents, self._tokenized):
            if types and doc.type not in types:
                continue
            score = self._score_doc(q_tokens, tokens)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)

        out: List[Dict] = []
        for score, doc in scored[:k]:
            out.append(
                {
                    "id": doc.id,
                    "type": doc.type,
                    "name": doc.name,
                    "text": doc.text,
                    "score": round(score, 3),
                }
            )
        return out

    def format_for_prompt(self, results: List[Dict]) -> str:
        """Render retrieval results as a compact bulleted block for prompts."""
        if not results:
            return "(no relevant context retrieved)"
        lines = []
        for r in results:
            lines.append(
                f"- [{r['type']}:{r['name']}] (score {r['score']}): {r['text']}"
            )
        return "\n".join(lines)
