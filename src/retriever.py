import re
from typing import Optional, Any
import bm25s
import numpy as np
from tqdm import tqdm
from src.ingester import Chunk, Ingester
from src.models import MinimalSource

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

        # Legacy - search_with_fallbackLE INDENT

        self.bm25_docs: Optional[bm25s.BM25] = None
        self.bm25_code: Optional[bm25s.BM25] = None
        self.doc_chunks: list[Chunk] = []
        self.code_chunks: list[Chunk] = []

        # Legacy - search_with_fallbackLE INDENT
        
        if embeddings is not None:
            self.load_embed_model()

    def search_with_type_fallback(
        self, query: str, k: int = 10
    ) -> list[MinimalSource]:
        """Search using type detection from first BM25 result.

        Gets top-1 from general index to detect query type (doc vs code),
        then searches the matching subindex for top-k results.
        Falls back to general search if subindices unavailable.

        Args:
            query: Natural-language search string.
            k: Number of results to return.

        Returns:
            Ranked list of MinimalSource objects.
        """
        if not query or not query.strip() or k <= 0:
            return []

        if self.bm25_docs is None or self.bm25_code is None:
            return self.search(query, k)

        expanded = expand_query(query)

        # Step 1 — get top-3 from general to detect type
        top3 = self.search_bm25(expanded, 3, min_score=0.0)
        if not top3:
            return self.search(query, k)

        chunk_map = self.get_chunk_map()
        doc_count = 0
        code_count = 0
        for source in top3:
            chunk = chunk_map.get(
                (source.file_path, source.first_character_index)
            )
            if chunk and chunk.chunk_type == "doc":
                doc_count += 1
            elif chunk and chunk.chunk_type == "code":
                code_count += 1

            if doc_count >= 2:
                subindex = self.bm25_docs
                sub_chunks = self.doc_chunks
            else:
                subindex = self.bm25_code
                sub_chunks = self.code_chunks
        
        # Step 2 — search in matching subindex
        try:
            tokenized = bm25s.tokenize(
                [expanded], stopwords="en", show_progress=False
            )
            effective_k = min(k, len(sub_chunks))
            results, _ = subindex.retrieve(
                tokenized, k=effective_k, show_progress=False
            )
            local_indices: list[int] = results[0].tolist()
        except Exception:
            return self.search(query, k)

        return [
            self.chunk_to_source(sub_chunks[int(i)])
            for i in local_indices
        ]

    def rerank_batch(self, queries: list[str],
                     sources_list: list[list[MinimalSource]],
                     k: int) -> list[list[MinimalSource]]:
        """Rerank candidates for multiple queries in a single batch call.

        More efficient than calling rerank() per query because the cross-encoder
        processes all pairs in one forward pass instead of N separate calls.

        Args:
            queries: List of original search strings.
            sources_list: Parallel list of candidate MinimalSource lists.
            k: Number of top results to return per query.

        Returns:
            List of reranked MinimalSource lists, one per query.
        """
        if self.reranker is None:
            return [s[:k] for s in sources_list]

        chunk_map = self.get_chunk_map()
        all_pairs: list[tuple[str, str]] = []
        offsets: list[int] = [0]

        for query, sources in zip(queries, sources_list):
            for source in sources:
                chunk = chunk_map.get(
                    (source.file_path, source.first_character_index)
                )
                text = chunk.text if chunk else source.file_path
                all_pairs.append((query, text))
            offsets.append(len(all_pairs))

        if not all_pairs:
            return [s[:k] for s in sources_list]

        all_scores: list[float] = self.reranker.predict(
            all_pairs, show_progress_bar=True).tolist()

        results: list[list[MinimalSource]] = []
        for i, sources in enumerate(sources_list):
            start, end = offsets[i], offsets[i + 1]
            query_scores = all_scores[start:end]
            ranked = sorted(
                zip(query_scores, sources),
                key=lambda x: x[0],
                reverse=True
            )
            results.append([s for _, s in ranked[:k]])

        return results

    def get_chunk_map(self) -> dict:
        """Return cached chunk map for fast lookup by (file_path, first_char)."""
        if not self.chunk_map:
            self.chunk_map = {
                (c.file_path, c.first_character_index): c
                for c in self.chunks
            }
        return self.chunk_map

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

        # Legacy - search_with_fallbackLE INDENT
        retriever.bm25_docs = ingester.bm25_docs
        retriever.bm25_code = ingester.bm25_code
        retriever.doc_chunks = [c for c in ingester.chunks if c.chunk_type == "doc"]
        retriever.code_chunks = [c for c in ingester.chunks if c.chunk_type == "code"]
        # Legacy - search_with_fallbackLE INDENT

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

    def search_for_generation(self, query: str,
                              k: int = 10) -> list[MinimalSource]:
        """Precise retrieval for LLM generation — BM25 + reranker.

        Retrieves k*5 candidates with BM25 then reranks them with the
        cross-encoder to surface the most relevant chunks for the LLM.
        Falls back to search() if the reranker is not loaded.

        Args:
            query: Natural-language search string.
            k: Number of results to return after reranking.

        Returns:
            Reranked list of MinimalSource objects, best match first.
        """

        if not query or not query.strip() or k <= 0:
            return []
        
        if self.get_top_bm25_score(query) < 4.5:
            return []

        if self.reranker is None:
            return self.search(query, k)

        expanded = expand_query(query, self.chunks)
        fetch_k = min(k * 5, len(self.chunks))

        if self.embeddings is not None and self.embed_model is not None:
            candidates = self.search_hybrid(expanded, fetch_k)
        else:
            candidates = self.search_bm25(expanded, fetch_k, min_score=0.0)

        return self.rerank(query, candidates, k)

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

    def rerank(self, query: str, sources: list[MinimalSource],
               k: int) -> list[MinimalSource]:
        """Rerank candidates using a cross-encoder for precision.

        The cross-encoder scores each (query, chunk_text) pair jointly,
        producing a relevance score more accurate than BM25 alone.

        Args:
            query: The original search string (not expanded).
            sources: Candidate MinimalSource objects to rerank.
            k: Number of top results to return after reranking.

        Returns:
            Reranked and truncated MinimalSource list.
        """
        if self.reranker is None or not sources:
            return sources[:k]

        chunk_map = self.get_chunk_map()

        pairs: list[tuple[str, str]] = []
        for source in sources:
            chunk = chunk_map.get(
                (source.file_path, source.first_character_index)
            )
            text = chunk.text if chunk else source.file_path
            pairs.append((query, text))

        scores: list[float] = self.reranker.predict(pairs).tolist()
        ranked = sorted(
            zip(scores, sources), key=lambda x: x[0], reverse=True
        )
        return [source for _, source in ranked[:k]]

    def retrieve_from_bm25(self, query: str, k: int) -> Any | list[int]:
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

