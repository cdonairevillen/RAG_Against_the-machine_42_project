import re
from typing import Optional
import bm25s
import numpy as np
from src.ingester import Chunk, Ingester
from src.models import MinimalSource

RRF_K = 60


def expand_query(query: str, chunks: Optional[list] = None) -> str:
    """Expand a query with identifier variants to improve BM25 recall."""
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

    Provides three search modes:
        search()              — fast BM25-only, maps children to parents.
        search_smart()        — BM25 for docs, reranker for code.
        search_dual()         — separate BM25 indices for docs and code,
                                interleaved. No reranker needed.
        search_for_generation() — BM25 + reranker, used before the LLM.

    Attributes:
        chunks: All indexed Chunk objects (children).
        parent_chunks: All parent Chunk objects.
        bm25: Fitted unified BM25 instance.
        bm25_code: Fitted BM25 instance for code chunks only.
        bm25_doc: Fitted BM25 instance for doc chunks only.
        code_chunk_indices: Indices into chunks[] for code chunks.
        doc_chunk_indices: Indices into chunks[] for doc chunks.
        embeddings: Dense vectors for all chunks, or None.
        embed_model: Loaded SentenceTransformer, or None.
        reranker: Loaded CrossEncoder, or None.
    """

    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, chunks: list[Chunk],
                 bm25: bm25s.BM25,
                 embeddings: Optional[np.ndarray] = None) -> None:
        self.chunks = chunks
        self.bm25 = bm25
        self.embeddings = embeddings
        self.embed_model = None
        self.reranker = None

        self.chunk_map: dict = {}
        self.parent_chunk_map: dict = {}
        self.parent_chunks: list[Chunk] = []

        self.bm25_code: Optional[bm25s.BM25] = None
        self.bm25_doc: Optional[bm25s.BM25] = None
        self.code_chunk_indices: list[int] = []
        self.doc_chunk_indices: list[int] = []

        if embeddings is not None:
            self.load_embed_model()

        self.doc_embeddings: Optional[np.ndarray] = None
        self.doc_parent_indices: list[int] = []

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

    def load_embed_model(self) -> None:
        """Load the sentence-transformers model for query encoding."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(self.EMBED_MODEL)
            print("Embedding model loaded — hybrid retrieval active.")
        except Exception as exc:
            print(f"Warning: embedding model unavailable ({exc}).")
            self.embeddings = None

    def load_reranker(self) -> None:
        """Load the cross-encoder reranker model."""
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.RERANKER_MODEL)
            print("Reranker loaded — precision reranking active.")
        except Exception as exc:
            print(f"Warning: reranker unavailable ({exc}).")
            self.reranker = None

    @classmethod
    def from_disk(cls, processed_dir: str) -> "Retriever":
        """Load a Retriever from a directory produced by Ingester.save()."""
        ingester = Ingester.load(processed_dir)
        if ingester.bm25 is None:
            raise RuntimeError("BM25 index failed to load from disk.")
        retriever = cls(
            chunks=ingester.chunks,
            bm25=ingester.bm25,
            embeddings=ingester.doc_embeddings,
        )
        retriever.parent_chunks = ingester.parent_chunks
        retriever.bm25_code = ingester.bm25_code
        retriever.bm25_doc = ingester.bm25_doc
        retriever.code_chunk_indices = ingester.code_chunk_indices
        retriever.doc_chunk_indices = ingester.doc_chunk_indices
        retriever.doc_embeddings = ingester.doc_embeddings
        retriever.doc_parent_indices = ingester.doc_parent_indices

        if ((retriever.doc_embeddings is not None) and (retriever.embed_model
                                                        is None)):
            retriever.load_embed_model()
        return retriever

    def search(self, query: str, k: int = 10) -> list[MinimalSource]:
        """Fast BM25-only retrieval mapping children to parents.

        Args:
            query: Natural-language search string.
            k: Number of results to return.

        Returns:
            Ranked list of parent MinimalSource objects.
        """
        if not query or not query.strip() or k <= 0:
            return []

        expanded = expand_query(query, self.chunks)

        if self.embeddings is not None and self.embed_model is not None:
            return self.search_hybrid(expanded, k)
        return self.search_bm25(expanded, k, min_score=0.0)

    def search_no_rerank(self, query: str, k: int = 10) -> list[MinimalSource]:
        """BM25 retrieval mapping children to parents, no reranker.

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
        child_candidates = self.search_bm25(expanded, fetch_k, min_score=0.0)

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
            key = (parent_source.file_path,
                   parent_source.first_character_index)
            if key not in seen:
                seen.add(key)
                parent_sources.append(parent_source)
            if len(parent_sources) >= k:
                break

        return parent_sources

    def rerank_by_embeddings(self, query: str,
                             sources: list[MinimalSource],
                             k: int) -> list[MinimalSource]:
        """Reorder doc sources by semantic similarity."""
        if ((not sources or self.doc_embeddings is None) or (self.embed_model
                                                             is None)):
            return sources[:k]

        embed_idx = {
            (self.parent_chunks[i].file_path,
             self.parent_chunks[i].first_character_index): rank
            for rank, i in enumerate(self.doc_parent_indices)
            if i < len(self.parent_chunks)
        }

        query_vec = self.embed_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
        )

        scored: list[tuple[float, MinimalSource]] = []
        for src in sources:
            key = (src.file_path, src.first_character_index)
            pidx = embed_idx.get(key)
            if pidx is None or pidx >= len(self.doc_embeddings):
                scored.append((0.0, src))
                continue
            score = float((self.doc_embeddings[pidx] @ query_vec.T).squeeze())
            scored.append((score, src))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [src for _, src in scored[:k]]

    def search_smart(self, query: str, k: int = 10) -> list[MinimalSource]:
        """
        Classifier-guided retrieval — doc index for docs,
        reranker for code.
        """
        if not query or not query.strip() or k <= 0:
            return []

        expanded = expand_query(query, self.chunks)

        # Clasificador: comparar scores top-3 de cada índice + señales léxicas
        doc_top_score = 0.0
        code_top_score = 0.0
        try:
            tokenized = bm25s.tokenize([expanded], stopwords="en",
                                       show_progress=False)
            _, ds = self.bm25_doc.retrieve(tokenized, k=3,
                                           show_progress=False)
            doc_top_score = float(np.mean(ds[0]))
            _, cs = self.bm25_code.retrieve(tokenized, k=3,
                                            show_progress=False)
            code_top_score = float(np.mean(cs[0]))
        except Exception:
            pass

        query_terms = set(expanded.lower().split())
        code_term_signals = sum(1 for t in query_terms if (
            '_' in t or
            (t[0].isupper() if t else False) or
            t in {'class', 'def', 'return', 'attribute',
                  'instance', 'import', 'module', 'init',
                  'self'}
        ))
        doc_term_signals = sum(1 for t in query_terms if t in {
            'how', 'what', 'when', 'where', 'why', 'configure',
            'install', 'setup', 'guide', 'documentation', 'example',
            'tutorial', 'start', 'run', 'serve', 'deploy', 'support',
            'compatible', 'hardware', 'version', 'requirement',
            'platform', 'gpu', 'rocm', 'amd', 'cli', 'command',
            'api', 'endpoint', 'port', 'package', 'install',
            'capability', 'compute', 'status', 'feature'
        })

        combined_code = code_top_score + code_term_signals * 0.5
        combined_doc = doc_top_score + doc_term_signals * 0.5
        MARGIN = 0.5

        if combined_code > combined_doc + MARGIN:
            # Modo código: BM25 unificado filtrado + reranker
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
                key = (parent_source.file_path,
                       parent_source.first_character_index)
                if key not in seen:
                    seen.add(key)
                    code_sources.append(parent_source)
                if len(code_sources) >= k * 3:
                    break

            # Añadir 2 docs del índice de docs como seguridad
            doc_backup: list[MinimalSource] = []
            if self.bm25_doc is not None and self.doc_chunk_indices:
                try:
                    tokenized = bm25s.tokenize([expanded], stopwords="en",
                                               show_progress=False)
                    results, _ = self.bm25_doc.retrieve(tokenized, k=10,
                                                        show_progress=False)
                    seen_d: set = set()
                    for idx in results[0].tolist():
                        chunk = self.chunks[self.doc_chunk_indices[int(idx)]]
                        ps = self.child_to_parent_source(chunk)
                        key = (ps.file_path, ps.first_character_index)
                        if key not in seen_d and key not in seen:
                            seen_d.add(key)
                            doc_backup.append(ps)
                        if len(doc_backup) >= 4:
                            break
                except Exception:
                    pass

            # Reranker solo si hay ambigüedad — si code_top_score > 12,
            # BM25 directo
            if code_top_score > 9.0:
                return code_sources[:k]
            return self.rerank_parents(query, code_sources + doc_backup, k)

        else:
            # Modo docs: índice de docs primero
            fetch_k = min(k * 5, len(self.doc_chunk_indices))
            doc_sources: list[MinimalSource] = []
            if self.bm25_doc is not None and self.doc_chunk_indices:
                try:
                    tokenized = bm25s.tokenize([expanded], stopwords="en",
                                               show_progress=False)
                    results, _ = self.bm25_doc.retrieve(tokenized, k=fetch_k,
                                                        show_progress=False)
                    seen_d: set = set()
                    for idx in results[0].tolist():
                        chunk = self.chunks[self.doc_chunk_indices[int(idx)]]
                        ps = self.child_to_parent_source(chunk)
                        key = (ps.file_path, ps.first_character_index)
                        if key not in seen_d:
                            seen_d.add(key)
                            doc_sources.append(ps)
                except Exception:
                    pass

            # Si hay pocos docs, completar con el índice unificado
            if len(doc_sources) < k:
                unified = self.search_no_rerank(query, k * 2)
                seen_u = {(s.file_path, s.first_character_index)
                          for s in doc_sources}
                for src in unified:
                    if len(doc_sources) >= k * 2:
                        break
                    key = (src.file_path, src.first_character_index)
                    if key not in seen_u:
                        seen_u.add(key)
                        doc_sources.append(src)

            # Reordenar por embeddings si disponibles
            if ((self.doc_embeddings is not None) and (self.embed_model
                                                       is not None)):
                return self.rerank_by_embeddings(query, doc_sources, k)
            return doc_sources[:k]

    def get_top_bm25_score(self, query: str) -> float:
        """Return the highest BM25 score for a query over the corpus."""
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
        """Rerank parent sources using parent chunk text."""
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
            return self.search_no_rerank(query, k)

        expanded = expand_query(query, self.chunks)
        fetch_k = min(k * 5, len(self.chunks))

        if self.embeddings is not None and self.embed_model is not None:
            child_candidates = self.search_hybrid(expanded, fetch_k)
        else:
            child_candidates = self.search_bm25(expanded, fetch_k,
                                                min_score=0.0)

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

        all_sources = doc_sources + code_sources
        reranked = self.rerank_parents(query, all_sources,
                                       k + min(2, len(doc_sources)))

        result: list[MinimalSource] = []
        seen_result: set = set()
        min_docs = min(2, len(doc_sources))

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
        """BM25-only retrieval over the unified index."""
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
        """Hybrid retrieval: BM25 + semantic embeddings fused with RRF."""
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
        """Run a BM25 query and return chunk indices."""
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
        """Return top-k chunk indices by cosine similarity."""
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
        """Convert a Chunk to a MinimalSource."""
        return MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )

    def child_to_parent_source(self, child: Chunk) -> MinimalSource:
        """Map a child chunk to its parent's MinimalSource."""
        if child.parent_id >= 0 and child.parent_id < len(self.parent_chunks):
            parent = self.parent_chunks[child.parent_id]
            return MinimalSource(
                file_path=parent.file_path,
                first_character_index=parent.first_character_index,
                last_character_index=parent.last_character_index)
        return self.chunk_to_source(child)
