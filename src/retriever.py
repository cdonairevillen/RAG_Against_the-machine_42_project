import bm25s
from tqdm import tqdm

from src.ingester import Chunk, Ingester
from src.models import MinimalSource


class Retriever:
    """Searches the indexed knowledge base and returns ranked source chunks.

    Wraps a fitted BM25 index and the chunk list produced by Ingester.
    Exposes a clean search() interface that returns MinimalSource objects
    directly usable by the CLI and the Evaluator.

    Usage:
        retriever = Retriever.from_disk("data/processed")
        sources = retriever.search("How does PagedAttention work?", k=10)

    Attributes:
        chunks: All indexed Chunk objects loaded from disk.
        bm25: Fitted bm25s.BM25 instance.
    """

    def __init__(self, chunks: list[Chunk], bm25: bm25s.BM25) -> None:
        """Initialize with already-loaded chunks and BM25 index.

        Prefer Retriever.from_disk() for normal usage.

        Args:
            chunks: List of Chunk objects matching the BM25 index order.
            bm25: Fitted bm25s.BM25 instance.
        """
        self.chunks = chunks
        self.bm25 = bm25

    @classmethod
    def from_disk(cls, processed_dir: str) -> "Retriever":
        """Load a Retriever from a directory produced by Ingester.save().

        Args:
            processed_dir: Path to the directory containing chunks/ and
                bm25_index/ subdirectories.

        Returns:
            A ready-to-use Retriever instance.

        Raises:
            RuntimeError: If the BM25 index fails to load.
        """
        ingester = Ingester.load(processed_dir)
        if ingester.retriever is None:
            raise RuntimeError("BM25 index failed to load from disk.")
        return cls(chunks=ingester.chunks, bm25=ingester.retriever)

    def search(self, query: str, k: int = 10) -> list[MinimalSource]:
        """Return the top-k most relevant chunks for a single query.

        Empty or whitespace-only queries and k=0 return an empty list
        cleanly, satisfying the edge-case requirements.

        Args:
            query: Natural-language search string.
            k: Number of results to return.

        Returns:
            Ranked list of MinimalSource objects (best match first).
        """
        if not query or not query.strip() or k <= 0:
            return []

        try:
            tokenized = bm25s.tokenize(
                [query], stopwords="en", show_progress=False
            )
            effective_k = min(k, len(self.chunks))
            results, _ = self.bm25.retrieve(
                tokenized, k=effective_k, show_progress=False
            )
            indices: list[int] = results[0].tolist()
        except Exception:
            return []

        return [
            MinimalSource(
                file_path=self.chunks[idx].file_path,
                first_character_index=self.chunks[idx].first_character_index,
                last_character_index=self.chunks[idx].last_character_index,
            )
            for idx in indices
        ]

    def search_batch(self, questions: list[str],
                     k: int = 10) -> list[list[MinimalSource]]:
        """Run search() over a list of questions with a progress bar.

        Args:
            questions: List of natural-language question strings.
            k: Number of results per question.

        Returns:
            List of MinimalSource lists, one per question, same order.
        """
        return [
            self.search(q, k=k)
            for q in tqdm(questions, desc="Searching")
        ]

    @staticmethod
    def _overlap_ratio(gt: tuple[str, int, int],
                       ret: tuple[str, int, int]) -> float:
        """Return the fraction of the ground-truth range covered by a
        retrieved chunk.

        Args:
            gt: (file_path, first_char, last_char) ground-truth source.
            ret: (file_path, first_char, last_char) retrieved source.

        Returns:
            Float in [0.0, 1.0]. 0.0 if files differ or no overlap.
        """
        gt_file, gt_start, gt_end = gt
        ret_file, ret_start, ret_end = ret

        if gt_file != ret_file:
            return 0.0

        overlap = max(0, min(gt_end, ret_end) - max(gt_start, ret_start))
        gt_length = gt_end - gt_start
        return overlap / gt_length if gt_length > 0 else 0.0
