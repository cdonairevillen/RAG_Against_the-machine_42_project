from typing import Any
from src.models import (StudentSearchResults,
                        RagDataset, AnsweredQuestion)


class Evaluator:
    """Measures retrieval quality using Recall@k against ground-truth sources.

    A retrieved source counts as a hit when it overlaps at least 5% with
    any ground-truth source, measured in characters over ground-truth length:

        overlap_chars / gt_length >= 0.05  ->  hit

    For questions with multiple correct sources, each is scored independently
    and the question score is found / total (partial credit).

    Usage:
        evaluator = Evaluator()
        metrics = evaluator.compute_recall(
            student_path="data/output/search_results/dataset_docs_public.json",
            ground_truth_path=(
                "data/datasets/AnsweredQuestions/dataset_docs_public.json"
            ),
            k=10,
        )
        evaluator.print_report(metrics)
    """

    def compute_recall(self, student_path: str, ground_truth_path: str,
                       k: int = 10) -> dict[str, Any]:
        """Compute Recall@1, @3, @5 and @k against ground-truth sources.

        Args:
            student_path: Path to StudentSearchResults JSON.
            ground_truth_path: Path to AnsweredQuestions RagDataset JSON.
            k: Upper k cutoff. Always evaluates @1, @3, @5 plus @k.

        Returns:
            Dict with:
                "total": int — questions evaluated.
                "recall_at_k": dict[int, float] — recall per k cutoff.
        """
        with open(student_path, "r", encoding="utf-8") as fh:
            st_dt = StudentSearchResults.model_validate_json(fh.read())

        with open(ground_truth_path, "r", encoding="utf-8") as fh:
            gt_data = RagDataset.model_validate_json(fh.read())

        gt_lookup = self.build_gt_lookup(gt_data)
        k_values = sorted({1, 3, 5, k})
        recall_sums: dict[int, float] = {kv: 0.0 for kv in k_values}
        evaluated = 0

        for entry in st_dt.search_results:
            gt_sources = gt_lookup.get(entry.question_id)
            if not gt_sources:
                continue
            evaluated += 1

            retrieved = [
                (s.file_path, s.first_character_index, s.last_character_index)
                for s in entry.retrieved_sources
            ]

            for kv in k_values:
                top_k = retrieved[:kv]
                found = sum(
                    1 for gt in gt_sources
                    if any(
                        self.overlap_ratio(gt, ret) >= 0.05
                        for ret in top_k
                    )
                )
                recall_sums[kv] += found / len(gt_sources)

        recall_at_k = {
            kv: (recall_sums[kv] / evaluated if evaluated > 0 else 0.0)
            for kv in k_values
        }
        return {"total": evaluated, "recall_at_k": recall_at_k}

    def print_report(self, metrics: dict[str, Any]) -> None:
        """Print a formatted evaluation report to stdout.

        Args:
            metrics: Dict returned by compute_recall().
        """
        print("\nEvaluation Results")
        print("=" * 40)
        print(f"Questions evaluated: {metrics['total']}")
        for ki, score in metrics["recall_at_k"].items():
            bar = "█" * int(score * 20)
            print(f"Recall@{ki:2d}: {score:.3f}  {bar}")

    def build_gt_lookup(self, gt_data: RagDataset
                        ) -> dict[str, list[tuple[str, int, int]]]:
        """Build a question_id to source ranges lookup from a RagDataset.

        Args:
            gt_data: Ground-truth RagDataset (AnsweredQuestions).

        Returns:
            Dict mapping question_id to list of (file_path, start, end).
        """
        lookup: dict[str, list[tuple[str, int, int]]] = {}
        for q in gt_data.rag_questions:
            if isinstance(q, AnsweredQuestion):
                lookup[q.question_id] = [
                    (
                        s.file_path,
                        s.first_character_index,
                        s.last_character_index,
                    )
                    for s in q.sources
                ]
        return lookup

    def overlap_ratio(self, gt: tuple[str, int, int],
                      ret: tuple[str, int, int]) -> float:
        """Return fraction of the ground-truth range covered by a retrieved
        chunk.

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
