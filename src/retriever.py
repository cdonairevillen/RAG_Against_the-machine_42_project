import re
from typing import Optional
import bm25s
import numpy as np
from src.ingester import Chunk, Ingester
from src.models import MinimalSource
from pathlib import Path

RRF_K = 60


def expand_query(query: str, chunks: Optional[list] = None) -> str:
    """Expand a query with identifier variants to improve BM25 recall.

    Detects CamelCase class names and snake_case identifiers in the query
    and adds decomposed variants. When chunks are provided, also boosts
    terms found in Chunk.symbols that match query tokens.

    Args:
        query: The original natural-language search string.
        chunks: Optional list of Chunk objects. When provided, symbol names
            matching query tokens are appended to the expanded query.

    Returns:
        Expanded query string with additional identifier variants.
    """
    extras: list[str] = []

    camel_pattern = re.compile(
        r'\b([A-Z][a-zA-Z0-9]*(?:[A-Z][a-z0-9]+)+)\b'
    )
    for match in camel_pattern.finditer(query):
        term = match.group(1)
        parts = re.sub(r'([A-Z])', r' \1', term).strip().split()
        if len(parts) > 1:
            extras.extend(parts)
            extras.append("_".join(p.lower() for p in parts))

    snake_pattern = re.compile(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+){2,})\b')
    for match in snake_pattern.finditer(query):
        term = match.group(1)
        parts = term.split("_")
        extras.extend(parts)

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
    """Searches the indexed knowledge base using BM25 with query expansion.

    Provides two search modes:
        search() — fast BM25-only, used by search_dataset for evaluation.
        search_for_generation() — BM25 + reranker, used before the LLM.

    The reranker is loaded lazily via load_reranker() so that search_dataset
    does not pay the model loading cost.

    Usage:
        retriever = Retriever.from_disk("data/processed")

        # Fast path — for evaluation
        sources = retriever.search(query, k=10)

        # Precise path — for LLM generation
        retriever.load_reranker()
        sources = retriever.search_for_generation(query, k=10)

    Attributes:
        chunks: All indexed Chunk objects.
        bm25: Fitted unified BM25 instance.
        embeddings: Dense vectors for all chunks, or None.
        embed_model: Loaded SentenceTransformer, or None.
        reranker: Loaded CrossEncoder, or None.
    """

    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, chunks: list[Chunk],
                 bm25: bm25s.BM25,
                 embeddings: Optional[np.ndarray] = None) -> None:
        """Initialise with pre-loaded components.

        Prefer Retriever.from_disk() for normal usage.

        Args:
            chunks: All Chunk objects in corpus order.
            bm25: Fitted unified BM25 instance.
            embeddings: Optional array of shape (n_chunks, embed_dim).
        """
        self.chunks = chunks
        self.bm25 = bm25
        self.embeddings = embeddings
        self.embed_model = None
        self.reranker = None

        self.chunk_map: dict = {}
        self.parent_chunk_map: dict = {}
        self.parent_chunks: list[Chunk] = []

        if embeddings is not None:
            self.load_embed_model()

    def get_chunk_map(self) -> dict:
        """
        Return cached chunk map for fast lookup by (file_path, first_char).
        """
        if not self.chunk_map:
            self.chunk_map = {
                (c.file_path, c.first_character_index): c
                for c in self.chunks
            }
        return self.chunk_map

    def get_parent_chunk_map(self) -> dict:
        """Return cached parent chunk map for fast lookup."""
        if not self.parent_chunk_map:
            self.parent_chunk_map = {
                (c.file_path, c.first_character_index): c
                for c in self.parent_chunks
            }
        return self.parent_chunk_map

    def load_embed_model(self) -> None:
        """Load the sentence-transformers model for query encoding.

        Falls back gracefully if the model cannot be loaded.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(self.EMBED_MODEL)
            print("Embedding model loaded — hybrid retrieval active.")
        except Exception as exc:
            print(f"Warning: embedding model unavailable ({exc}).")
            self.embeddings = None

    def load_reranker(self) -> None:
        """Load the cross-encoder reranker model.

        Call this explicitly before search_for_generation() when
        precision matters more than speed. Falls back gracefully.
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
            processed_dir: Path containing chunks/ and bm25_index/.

        Returns:
            A ready-to-use Retriever instance.

        Raises:
            RuntimeError: If the BM25 index fails to load.
        """
        ingester = Ingester.load(processed_dir)
        if ingester.bm25 is None:
            raise RuntimeError("BM25 index failed to load from disk.")
        retriever = cls(
            chunks=ingester.chunks,
            bm25=ingester.bm25,
            embeddings=ingester.embeddings,
        )

        retriever.parent_chunks = ingester.parent_chunks

        return retriever

    def search(self, query: str, k: int = 10) -> list[MinimalSource]:
        """Fast retrieval for evaluation — BM25 with query expansion only.

        No reranking. Optimised for throughput over the full dataset.

        Args:
            query: Natural-language search string.
            k: Number of results to return.

        Returns:
            Ranked list of MinimalSource objects, best match first.
        """
        if not query or not query.strip() or k <= 0:
            return []

        expanded = expand_query(query, self.chunks)

        if self.embeddings is not None and self.embed_model is not None:
            return self.search_hybrid(expanded, k)
        return self.search_bm25(expanded, k, min_score=0.0)

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

    def search_smart(self, query: str, k: int = 10) -> list[MinimalSource]:
        """BM25 para docs, reranker para code."""
        if not query or not query.strip() or k <= 0:
            return []

        expanded = expand_query(query, self.chunks)
        fetch_k = min(k * 5, len(self.chunks))
        child_candidates = self.search_bm25(expanded, fetch_k, min_score=0.0)

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
            key = (parent_source.file_path,
                   parent_source.first_character_index)
            if key not in seen:
                seen.add(key)
                if child.chunk_type == "doc":
                    doc_sources.append(parent_source)
                else:
                    code_sources.append(parent_source)

        # Docs: BM25 order (no reranker)
        # Code: reranker
        if self.reranker is not None and code_sources:
            top_code = self.rerank_parents(query, code_sources, k)
        else:
            top_code = code_sources[:k]

        # Garantizar representación de ambos tipos
        # Ajustar doc_slots según lo que BM25 encontró naturalmente
        natural_docs = len(doc_sources)
        if natural_docs == 0:
            doc_slots = 0
        elif natural_docs <= 2:
            doc_slots = natural_docs  # no forzar más de lo que hay
        else:
            doc_slots = min(3, natural_docs)

        code_slots = k - doc_slots

        result: list[MinimalSource] = []
        seen_result: set = set()

        # Primero los mejores docs (BM25 order)
        for src in doc_sources[:doc_slots]:
            key = (src.file_path, src.first_character_index)
            if key not in seen_result:
                result.append(src)
                seen_result.add(key)

        # Luego el mejor código (reranked)
        for src in top_code[:code_slots]:
            key = (src.file_path, src.first_character_index)
            if key not in seen_result:
                result.append(src)
                seen_result.add(key)

        # Rellenar si faltan slots
        for src in doc_sources[doc_slots:] + top_code[code_slots:]:
            if len(result) >= k:
                break
            key = (src.file_path, src.first_character_index)
            if key not in seen_result:
                result.append(src)
                seen_result.add(key)

        return result[:k]

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

        scores: list[float] = self.reranker.predict(pairs).tolist()
        ranked = sorted(
            zip(scores, sources), key=lambda x: x[0], reverse=True
        )
        return [source for _, source in ranked[:k]]

    def search_for_generation(self, query: str,
                              k: int = 10) -> list[MinimalSource]:
        """Precise retrieval for LLM generation — children for BM25,
        parents for reranking and context.

        Retrieves k*5 child candidates, maps to parent chunks,
        then reranks parents with the cross-encoder.

        Args:
            query: Natural-language search string.
            k: Number of results to return after reranking.

        Returns:
            Reranked list of parent MinimalSource objects.
        """
        if not query or not query.strip() or k <= 0:
            return []

        if self.get_top_bm25_score(query) < 5.5:
            return []

        if self.reranker is None:
            return self.search(query, k)

        expanded = expand_query(query, self.chunks)
        fetch_k = min(k * 5, len(self.chunks))

        # Retrieve children
        if self.embeddings is not None and self.embed_model is not None:
            child_candidates = self.search_hybrid(expanded, fetch_k)
        else:
            child_candidates = self.search_bm25(expanded, fetch_k,
                                                min_score=0.0)

        # Map children to parents, deduplicate
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
            key = (parent_source.file_path,
                   parent_source.first_character_index)
            if key not in seen:
                seen.add(key)
                if child.chunk_type == "doc":
                    doc_sources.append(parent_source)
                else:
                    code_sources.append(parent_source)

        # Rerank todo junto pero garantizando docs al frente
        all_sources = doc_sources + code_sources
        reranked = self.rerank_parents(query, all_sources, k +
                                       min(2, len(doc_sources)))

        # Garantizar al menos 2 docs en el resultado final
        result: list[MinimalSource] = []
        seen_result: set = set()
        min_docs = min(2, len(doc_sources))

        # Primero añadir docs del reranked
        for src in reranked:
            key = (src.file_path, src.first_character_index)
            if key not in seen_result:
                parent = self.get_parent_chunk_map().get(key)
                if parent and any(
                    (d.file_path, d.first_character_index) == key
                    for d in doc_sources
                ):
                    result.append(src)
                    seen_result.add(key)
                    if len(result) >= min_docs:
                        break

        # Rellenar con el resto del reranked
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

        return [self.chunk_to_source(self.chunks[int(idx)])
                for idx, score in zip(indices, result_scores)
                if score >= min_score]

    def search_hybrid(self, query: str, k: int) -> list[MinimalSource]:
        """Hybrid retrieval: BM25 + semantic embeddings fused with RRF.

        Args:
            query: Search string (already expanded).
            k: Final number of results.

        Returns:
            Ranked MinimalSource list.
        """

        fetch_k = min(k * 2, len(self.chunks))
        rrf_scores: dict[int, float] = {}

        bm25_indices = self.retrieve_from_bm25(query, fetch_k)
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = (
                rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
            )

        embed_indices = self.retrieve_from_embeddings(query, fetch_k)
        for rank, idx in enumerate(embed_indices):
            rrf_scores[idx] = (
                rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
            )

        ranked = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [
            self.chunk_to_source(self.chunks[idx])
            for idx, _ in ranked[:k]
        ]

    def retrieve_from_bm25(self, query: str, k: int) -> list[int]:
        """Run a BM25 query and return chunk indices.

        Args:
            query: Search string.
            k: Number of results.

        Returns:
            List of chunk indices ordered by descending BM25 score.
        """
        try:
            tokenized = bm25s.tokenize(
                [query], stopwords="en", show_progress=False
            )
            results, _ = self.bm25.retrieve(
                tokenized, k=k, show_progress=False
            )
            return results[0].tolist()
        except Exception:
            return []

    def retrieve_from_embeddings(self, query: str, k: int) -> list[int]:
        """Return top-k chunk indices by cosine similarity.

        Args:
            query: Search string.
            k: Number of results.

        Returns:
            List of chunk indices ordered by descending similarity.
        """
        if self.embed_model is None or self.embeddings is None:
            return []

        query_vec: np.ndarray = self.embed_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores = (self.embeddings @ query_vec.T).squeeze()
        effective_k = min(k, len(self.chunks))
        top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return top_indices.tolist()

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
            MinimalSource of the parent chunk, or child's own source
            if no parent exists.
        """
        if child.parent_id >= 0 and child.parent_id < len(self.parent_chunks):
            parent = self.parent_chunks[child.parent_id]
            return MinimalSource(
                file_path=parent.file_path,
                first_character_index=parent.first_character_index,
                last_character_index=parent.last_character_index)
        return self.chunk_to_source(child)
