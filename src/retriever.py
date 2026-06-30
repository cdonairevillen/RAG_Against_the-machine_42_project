import re
from typing import Optional
import bm25s
import numpy as np
from src.ingester import Chunk, Ingester
from src.models import MinimalSource

RRF_K = 60

CODE_SIGNAL_TERMS = {
    "def", "return", "attribute", "instance",
    "import", "module", "init", "self",
}

DOC_SIGNAL_TERMS = {
    "how", "what", "when", "where", "why", "configure",
    "install", "setup", "guide", "documentation", "example",
    "tutorial", "start", "run", "serve", "deploy", "support",
    "compatible", "hardware", "version", "requirement",
    "platform", "gpu", "rocm", "amd", "cli", "command",
    "api", "endpoint", "port", "package",
    "capability", "compute", "status", "feature",
}

CLASSIFIER_MARGIN = 0.5
CODE_SIGNAL_WEIGHT = 0.5
DOC_SIGNAL_WEIGHT = 0.5
SKIP_RERANK_SCORE = 9.0
BM25_SCORE_FLOOR = 5.5


def expand_query(query: str, chunks: Optional[list] = None) -> str:
    """Expand a query with identifier variants to improve BM25 recall.

    Detects CamelCase class names and snake_case identifiers in the
    query and adds decomposed variants. When chunks are provided,
    also boosts terms found in Chunk.symbols that match query tokens.

    Args:
        query: The original natural-language search string.
        chunks: Optional list of Chunk objects. When provided, symbol
            names matching query tokens are appended to the query.

    Returns:
        Expanded query string with additional identifier variants.
    """
    extras: list[str] = []

    camel_pattern = re.compile(
        r"\b([A-Z][a-zA-Z0-9]*(?:[A-Z][a-z0-9]+)+)\b"
    )
    for match in camel_pattern.finditer(query):
        term = match.group(1)
        parts = re.sub(r"([A-Z])", r" \1", term).strip().split()
        if len(parts) > 1:
            extras.extend(parts)
            extras.append("_".join(p.lower() for p in parts))

    snake_pattern = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+){2,})\b")
    for match in snake_pattern.finditer(query):
        term = match.group(1)
        extras.extend(term.split("_"))

    if chunks:
        query_tokens = set(query.lower().split())
        for chunk in chunks:
            if not chunk.symbols:
                continue
            for sym in chunk.symbols.split():
                if sym.lower() in query_tokens and sym not in extras:
                    extras.append(sym)

    if not extras:
        return query

    unique_extras = list(dict.fromkeys(
        e for e in extras if e.lower() not in query.lower()
    ))
    return query + " " + " ".join(unique_extras)


class Retriever:
    """Searches the indexed knowledge base using dual BM25 indices.

    Provides several search modes:
        search()                — fast BM25-only, maps children to
                                   parents.
        search_no_rerank()      — same as search(), explicit name.
        search_smart()          — classifier-guided: BM25 doc index
                                   for doc queries, reranker-filtered
                                   code index for code queries. The
                                   default and best-performing mode.
        search_for_generation() — BM25 + reranker over the unified
                                   index, used before the LLM.

    Attributes:
        chunks: All indexed child Chunk objects.
        parent_chunks: All parent Chunk objects.
        bm25: Fitted unified BM25 instance over all child chunks.
        bm25_code: Fitted BM25 instance over code child chunks only.
        bm25_doc: Fitted BM25 instance over doc child chunks only.
        code_chunk_indices: Indices into chunks[] for code chunks.
        doc_chunk_indices: Indices into chunks[] for doc chunks.
        reranker: Loaded CrossEncoder, or None until load_reranker().
    """

    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, chunks: list[Chunk],
                 bm25: bm25s.BM25) -> None:
        """Initialise with pre-loaded components.

        Prefer Retriever.from_disk() for normal usage.

        Args:
            chunks: All child Chunk objects in corpus order.
            bm25: Fitted unified BM25 instance.
        """
        self.chunks = chunks
        self.bm25 = bm25
        self.reranker = None

        self.chunk_map: dict = {}
        self.parent_chunk_map: dict = {}
        self.parent_chunks: list[Chunk] = []

        self.bm25_code: Optional[bm25s.BM25] = None
        self.bm25_doc: Optional[bm25s.BM25] = None
        self.code_chunk_indices: list[int] = []
        self.doc_chunk_indices: list[int] = []

    def get_chunk_map(self) -> dict:
        """Return cached child chunk map keyed by (file_path, first_char)."""
        if not self.chunk_map:
            self.chunk_map = {
                (c.file_path, c.first_character_index): c
                for c in self.chunks
            }
        return self.chunk_map

    def get_parent_chunk_map(self) -> dict:
        """Return cached parent chunk map keyed by (file_path, first_char)."""
        if not self.parent_chunk_map:
            self.parent_chunk_map = {
                (c.file_path, c.first_character_index): c
                for c in self.parent_chunks
            }
        return self.parent_chunk_map

    def load_reranker(self) -> None:
        """Load the cross-encoder reranker model.

        Call this explicitly before search_smart() or
        search_for_generation() when precision matters. Falls back
        gracefully if the model cannot be loaded.
        """
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.RERANKER_MODEL)
            print("Reranker loaded — precision reranking active.")
        except Exception as exc:
            print(f"Warning: reranker unavailable ({exc}).")
            self.reranker = None

    @classmethod
    def from_disk(cls, processed_dir: str) -> "Retriever":
        """Load a Retriever from a directory produced by Ingester.save().

        Args:
            processed_dir: Path containing chunks/ and the BM25
                index directories.

        Returns:
            A ready-to-use Retriever instance.

        Raises:
            RuntimeError: If the unified BM25 index fails to load.
        """
        ingester = Ingester.load(processed_dir)
        if ingester.bm25 is None:
            raise RuntimeError("BM25 index failed to load from disk.")
        retriever = cls(chunks=ingester.chunks, bm25=ingester.bm25)
        retriever.parent_chunks = ingester.parent_chunks
        retriever.bm25_code = ingester.bm25_code
        retriever.bm25_doc = ingester.bm25_doc
        retriever.code_chunk_indices = ingester.code_chunk_indices
        retriever.doc_chunk_indices = ingester.doc_chunk_indices
        return retriever

    def search(self, query: str, k: int = 10) -> list[MinimalSource]:
        """Fast BM25-only retrieval mapping children to parents.

        Args:
            query: Natural-language search string.
            k: Number of results to return.

        Returns:
            Ranked list of parent MinimalSource objects.
        """
        return self.search_no_rerank(query, k)

    def search_no_rerank(self, query: str,
                         k: int = 10) -> list[MinimalSource]:
        """BM25 retrieval over the unified index, mapped to parents.

        Args:
            query: Natural-language search string.
            k: Number of results to return.

        Returns:
            Ranked list of parent MinimalSource objects.
        """
        if not query or not query.strip() or k <= 0:
            return []

        expanded = expand_query(query, self.chunks)
        fetch_k = min(k * 5, len(self.chunks))
        child_candidates = self.search_bm25(
            expanded, fetch_k, min_score=0.0
        )

        parent_map = self.get_chunk_map()
        seen: set = set()
        parent_sources: list[MinimalSource] = []

        for source in child_candidates:
            child = parent_map.get(
                (source.file_path, source.first_character_index)
            )
            if child is None:
                continue
            parent_source = self.child_to_parent_source(child)
            key = (
                parent_source.file_path,
                parent_source.first_character_index,
            )
            if key not in seen:
                seen.add(key)
                parent_sources.append(parent_source)
            if len(parent_sources) >= k:
                break

        return parent_sources

    def classify_query(self, query: str) -> tuple[float, float]:
        """Score a query's affinity for the doc vs. code index.

        Combines the mean top-3 BM25 score from each per-type index
        with lexical signal counts (identifier-like tokens favour
        code, natural-language tokens favour docs).

        Args:
            query: The expanded search string.

        Returns:
            Tuple of (combined_code_score, combined_doc_score).
        """
        doc_top_score = 0.0
        code_top_score = 0.0
        try:
            tokenized = bm25s.tokenize(
                [query], stopwords="en", show_progress=False
            )
            if self.bm25_doc is not None:
                _, ds = self.bm25_doc.retrieve(
                    tokenized, k=3, show_progress=False
                )
                doc_top_score = float(np.mean(ds[0]))
            if self.bm25_code is not None:
                _, cs = self.bm25_code.retrieve(
                    tokenized, k=3, show_progress=False
                )
                code_top_score = float(np.mean(cs[0]))
        except Exception:
            pass

        query_terms = set(query.lower().split())
        code_signals = sum(
            1 for t in query_terms
            if "_" in t
            or (t[0].isupper() if t else False)
            or t in CODE_SIGNAL_TERMS
        )
        doc_signals = sum(
            1 for t in query_terms if t in DOC_SIGNAL_TERMS
        )

        combined_code = code_top_score + code_signals * CODE_SIGNAL_WEIGHT
        combined_doc = doc_top_score + doc_signals * DOC_SIGNAL_WEIGHT
        return combined_code, combined_doc

    def search_smart(self, query: str, k: int = 10) -> list[MinimalSource]:
        """Classifier-guided retrieval: doc index for docs, reranker
        for code.

        Routes each query to whichever per-type index it best
        matches. Doc queries are served by BM25 alone over the doc
        index (the reranker hurts doc ranking on this corpus). Code
        queries are served by the unified index filtered to code
        chunks, reranked unless the BM25 signal is already
        unambiguous — in which case the reranker is skipped to save
        time. A handful of doc results are always mixed into the
        code-mode pool as a safety net for borderline queries.

        Args:
            query: Natural-language search string.
            k: Number of results to return.

        Returns:
            Ranked list of parent MinimalSource objects.
        """
        if not query or not query.strip() or k <= 0:
            return []

        expanded = expand_query(query, self.chunks)
        combined_code, combined_doc = self.classify_query(expanded)

        if combined_code > combined_doc + CLASSIFIER_MARGIN:
            return self._search_smart_code(expanded, k, combined_code)
        return self._search_smart_docs(expanded, k)

    def _search_smart_code(self, expanded: str, k: int,
                           code_top_score: float) -> list[MinimalSource]:
        """Code-mode branch of search_smart(). See that method."""
        fetch_k = min(k * 10, len(self.chunks))
        candidates = self.search_bm25(expanded, fetch_k, min_score=0.0)
        chunk_map = self.get_chunk_map()

        code_sources: list[MinimalSource] = []
        seen: set = set()
        for source in candidates:
            child = chunk_map.get(
                (source.file_path, source.first_character_index)
            )
            if child is None or child.chunk_type != "code":
                continue
            parent_source = self.child_to_parent_source(child)
            key = (
                parent_source.file_path,
                parent_source.first_character_index,
            )
            if key not in seen:
                seen.add(key)
                code_sources.append(parent_source)
            if len(code_sources) >= k * 3:
                break

        doc_backup: list[MinimalSource] = []
        if self.bm25_doc is not None and self.doc_chunk_indices:
            try:
                tokenized = bm25s.tokenize(
                    [expanded], stopwords="en", show_progress=False
                )
                results, _ = self.bm25_doc.retrieve(
                    tokenized, k=10, show_progress=False
                )
                seen_doc: set = set()
                for idx in results[0].tolist():
                    chunk = self.chunks[self.doc_chunk_indices[int(idx)]]
                    ps = self.child_to_parent_source(chunk)
                    key = (ps.file_path, ps.first_character_index)
                    if key not in seen_doc and key not in seen:
                        seen_doc.add(key)
                        doc_backup.append(ps)
                    if len(doc_backup) >= 4:
                        break
            except Exception:
                pass

        if code_top_score > SKIP_RERANK_SCORE:
            return code_sources[:k]
        return self.rerank_parents(expanded, code_sources + doc_backup, k)

    def _search_smart_docs(self, expanded: str,
                           k: int) -> list[MinimalSource]:
        """Doc-mode branch of search_smart(). See that method."""
        fetch_k = min(k * 5, len(self.doc_chunk_indices))
        doc_sources: list[MinimalSource] = []

        if self.bm25_doc is not None and self.doc_chunk_indices:
            try:
                tokenized = bm25s.tokenize(
                    [expanded], stopwords="en", show_progress=False
                )
                results, _ = self.bm25_doc.retrieve(
                    tokenized, k=fetch_k, show_progress=False
                )
                seen: set = set()
                for idx in results[0].tolist():
                    chunk = self.chunks[self.doc_chunk_indices[int(idx)]]
                    ps = self.child_to_parent_source(chunk)
                    key = (ps.file_path, ps.first_character_index)
                    if key not in seen:
                        seen.add(key)
                        doc_sources.append(ps)
            except Exception:
                pass

        if len(doc_sources) < k:
            unified = self.search_no_rerank(expanded, k * 2)
            seen_unified = {
                (s.file_path, s.first_character_index)
                for s in doc_sources
            }
            for src in unified:
                if len(doc_sources) >= k * 2:
                    break
                key = (src.file_path, src.first_character_index)
                if key not in seen_unified:
                    seen_unified.add(key)
                    doc_sources.append(src)

        return doc_sources[:k]

    def get_top_bm25_score(self, query: str) -> float:
        """Return the highest BM25 score for a query over the corpus.

        Args:
            query: Search string.

        Returns:
            Highest score, or 0.0 on error.
        """
        try:
            tokenized = bm25s.tokenize(
                [query], stopwords="en", show_progress=False
            )
            _, scores = self.bm25.retrieve(
                tokenized, k=1, show_progress=False
            )
            return float(scores[0][0])
        except Exception:
            return 0.0

    def rerank_parents(self, query: str,
                       sources: list[MinimalSource],
                       k: int) -> list[MinimalSource]:
        """Rerank parent sources using parent chunk text.

        Args:
            query: The original search string.
            sources: Parent MinimalSource objects to rerank.
            k: Number of top results to return.

        Returns:
            Reranked and truncated MinimalSource list.
        """
        if self.reranker is None or not sources:
            return sources[:k]

        parent_map = self.get_parent_chunk_map()
        pairs: list[tuple[str, str]] = []
        for source in sources:
            parent = parent_map.get(
                (source.file_path, source.first_character_index)
            )
            text = parent.text if parent else source.file_path
            pairs.append((query, text))

        try:
            scores: list[float] = self.reranker.predict(pairs).tolist()
        except Exception:
            return sources[:k]

        ranked = sorted(
            zip(scores, sources), key=lambda x: x[0], reverse=True
        )
        return [source for _, source in ranked[:k]]

    def search_for_generation(self, query: str,
                              k: int = 10) -> list[MinimalSource]:
        """Precise retrieval for LLM generation — BM25 + reranker.

        Searches the unified index, splits candidates into doc and
        code parents, reranks them together, and guarantees at least
        two doc results survive into the final list when docs were
        present in the candidate pool.

        Args:
            query: Natural-language search string.
            k: Number of results to return after reranking.

        Returns:
            Reranked list of parent MinimalSource objects.
        """
        if not query or not query.strip() or k <= 0:
            return []

        if self.get_top_bm25_score(query) < BM25_SCORE_FLOOR:
            return []

        if self.reranker is None:
            return self.search_no_rerank(query, k)

        expanded = expand_query(query, self.chunks)
        fetch_k = min(k * 5, len(self.chunks))
        child_candidates = self.search_bm25(
            expanded, fetch_k, min_score=0.0
        )

        parent_map = self.get_chunk_map()
        seen: set = set()
        doc_sources: list[MinimalSource] = []
        code_sources: list[MinimalSource] = []

        for source in child_candidates:
            child = parent_map.get(
                (source.file_path, source.first_character_index)
            )
            if child is None:
                continue
            parent_source = self.child_to_parent_source(child)
            key = (
                parent_source.file_path,
                parent_source.first_character_index,
            )
            if key not in seen:
                seen.add(key)
                if child.chunk_type == "doc":
                    doc_sources.append(parent_source)
                else:
                    code_sources.append(parent_source)

        all_sources = doc_sources + code_sources
        reranked = self.rerank_parents(
            query, all_sources, k + min(2, len(doc_sources))
        )

        result: list[MinimalSource] = []
        seen_result: set = set()
        min_docs = min(2, len(doc_sources))
        doc_keys = {
            (d.file_path, d.first_character_index) for d in doc_sources
        }

        for src in reranked:
            key = (src.file_path, src.first_character_index)
            if key not in seen_result and key in doc_keys:
                result.append(src)
                seen_result.add(key)
                if len(result) >= min_docs:
                    break

        for src in reranked:
            key = (src.file_path, src.first_character_index)
            if key not in seen_result:
                result.append(src)
                seen_result.add(key)
            if len(result) >= k:
                break

        return result[:k]

    def search_bm25(self, query: str, k: int,
                    min_score: float = 0.0) -> list[MinimalSource]:
        """BM25-only retrieval over the unified index.

        Args:
            query: Search string (already expanded).
            k: Number of results.
            min_score: Minimum BM25 score to include a result.

        Returns:
            Ranked MinimalSource list.
        """
        try:
            tokenized = bm25s.tokenize(
                [query], stopwords="en", show_progress=False
            )
            effective_k = min(k, len(self.chunks))
            results, scores = self.bm25.retrieve(
                tokenized, k=effective_k, show_progress=False
            )
            indices: list[int] = results[0].tolist()
            result_scores: list[float] = scores[0].tolist()
        except Exception:
            return []

        return [
            self.chunk_to_source(self.chunks[int(idx)])
            for idx, score in zip(indices, result_scores)
            if score >= min_score
        ]

    def chunk_to_source(self, chunk: Chunk) -> MinimalSource:
        """Convert a Chunk to the MinimalSource model the CLI expects.

        Args:
            chunk: Internal Chunk dataclass.

        Returns:
            MinimalSource Pydantic model.
        """
        return MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )

    def child_to_parent_source(self, child: Chunk) -> MinimalSource:
        """Map a child chunk to its parent's MinimalSource.

        Args:
            child: Child Chunk with parent_id set.

        Returns:
            MinimalSource of the parent chunk, or the child's own
            source if it has no parent.
        """
        if 0 <= child.parent_id < len(self.parent_chunks):
            parent = self.parent_chunks[child.parent_id]
            return MinimalSource(
                file_path=parent.file_path,
                first_character_index=parent.first_character_index,
                last_character_index=parent.last_character_index,
            )
        return self.chunk_to_source(child)