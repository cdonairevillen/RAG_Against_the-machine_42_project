import os
from typing import Optional
from tqdm import tqdm
from src.models import (MinimalAnswer,
                        MinimalSearchResults,
                        RagDataset,
                        StudentSearchResults,
                        StudentSearchResultsAndAnswer)

PROCESSED_DIR = "data/processed"
REPO_ROOT = "data/raw/vllm-0.10.1"


class CLI:
    """RAG against the Machine — single orchestrator and CLI entry point.

    Owns one Retriever and one Generator instance, loaded lazily on first use
    and reused for all subsequent calls within the same process. This avoids
    reloading the BM25 index or the LLM weights between operations.

    Commands:
        index            Build the BM25 index from the vLLM repository.
        search           Search for a single query and print sources.
        search_dataset   Batch search over a full dataset JSON.
        answer           Answer a single question end-to-end.
        answer_dataset   Full pipeline: retrieve + generate over a dataset.
        evaluate         Compute Recall@k against ground truth.

    Examples:
        uv run python -m src index
        uv run python -m src search "How to configure OpenAI server?" --k 10
        uv run python -m src answer "How to configure OpenAI server?" --k 10
        uv run python -m src search_dataset \\
            --dataset_path data/datasets/UnansweredQuestions/
            dataset_docs_public.json
        uv run python -m src answer_dataset \\
            --dataset_path data/datasets/UnansweredQuestions/
            dataset_docs_public.json
        uv run python -m src evaluate \\
            --student_answer_path data/output/search_results/
            dataset_docs_public.json \\
            --dataset_path data/datasets/AnsweredQuestions/
            dataset_docs_public.json
    """

    def __init__(self, processed_dir: str = PROCESSED_DIR,
                 repo_root: str = REPO_ROOT) -> None:
        """Initialize the CLI orchestrator.

        Retriever and Generator are NOT loaded here — they are lazy-loaded
        on first use by _get_retriever() and _get_generator().

        Args:
            processed_dir: Directory containing chunks and BM25 index.
            repo_root: Root of the vLLM repository (used by Generator).
        """
        self._processed_dir = processed_dir
        self._repo_root = repo_root
        self._retriever: Optional[object] = None
        self._generator: Optional[object] = None

    def _get_retriever(self) -> "Retriever":  # type: ignore[name-defined]
        """Load and cache the Retriever. Subsequent calls return the same
        instance.

        Returns:
            The loaded Retriever instance.

        Raises:
            SystemExit: Prints an error and exits if the index is missing.
        """
        if self._retriever is None:
            from src.retriever import Retriever
            try:
                self._retriever = Retriever.from_disk(self._processed_dir)
            except Exception as e:
                print(f"Error loading index from '{self._processed_dir}': {e}")
                print("Have you run 'uv run python -m src index' yet?")
                raise
        return self._retriever  # type: ignore[return-value]

    def _get_generator(self) -> "Generator":
        """Load and cache the Generator (LLM). Subsequent calls return
        the same instance.

        Returns:
            The loaded Generator instance.

        Raises:
            Exception: Propagates any model loading error.
        """
        if self._generator is None:
            from src.generator import Generator
            try:
                self._generator = Generator(repo_root=self._repo_root)
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        return self._generator  # type: ignore[return-value]

    def index(self, repo_root: str = REPO_ROOT,
              max_chunk_size: int = 2000,
              output_dir: str = PROCESSED_DIR) -> None:
        """Ingest the vLLM repository and persist the BM25 index to disk.

        Args:
            repo_root: Path to the vLLM repository root.
            max_chunk_size: Maximum characters per chunk (configurable).
            output_dir: Where chunks and index are saved.
        """
        from src.ingester import Ingester

        if not os.path.isdir(repo_root):
            print(f"Error: repository not found at '{repo_root}'")
            print(f"Expected: {os.path.abspath(repo_root)}")
            return
        try:
            ingester = Ingester(repo_root=repo_root,
                                max_chunk_size=max_chunk_size)
            ingester.build()
            ingester.save(output_dir)
            print("Ingestion complete! Indices saved under", output_dir)
        except Exception as e:
            print(f"Error during indexing: {e}")

    def search(self, query: str, k: int = 10) -> None:
        """Search the knowledge base for a single query and print results.

        Args:
            query: Search string. Empty query exits cleanly without crashing.
            k: Number of results to retrieve.
        """
        if not query or not query.strip():
            print("Empty query — no results.")
            return
        try:
            retriever = self._get_retriever()
            results = retriever.search(query, k=k)
        except Exception as e:
            print(f"Error during search: {e}")
            return

        if not results:
            print("No results found.")
            return
        for i, source in enumerate(results, 1):
            print(f"[{i}] {source.file_path} "
                  f"({source.first_character_index}:"
                  f"{source.last_character_index})")

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/search_results"
                       ) -> None:
        """Run retrieval over every question in a dataset JSON and save
        results.

        Args:
            dataset_path: Path to a RagDataset JSON (answered or unanswered).
            k: Chunks to retrieve per question.
            save_directory: Where the SearchResults JSON is written.
        """
        if not os.path.isfile(dataset_path):
            print(f"Error: dataset file not found: {dataset_path}")
            return
        try:
            with open(dataset_path, "r", encoding="utf-8") as fh:
                dataset = RagDataset.model_validate_json(fh.read())
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return
        try:
            retriever = self._get_retriever()
        except Exception:
            return

        results: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            sources = retriever.search(question.question, k=k)
            results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources,
            ))

        self._save_json(
            StudentSearchResults(search_results=results, k=k),
            save_directory,
            os.path.basename(dataset_path),
            label="student_search_results",
        )

    def answer(self, query: str, k: int = 10) -> None:
        """Answer a single question using retrieved context and print the
        result.

        Args:
            query: Natural-language question.
            k: Number of chunks to retrieve as context.
        """
        if not query or not query.strip():
            print("Empty query — nothing to answer.")
            return
        try:
            retriever = self._get_retriever()
            sources = retriever.search(query, k=k)
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return

        if not sources:
            print("No relevant sources found.")
            return

        try:
            generator = self._get_generator()
            response = generator.answer(query, sources)
        except Exception as e:
            print(f"Error during generation: {e}")
            return

        print("\n=== Answer ===")
        print(response)
        print("\n=== Sources ===")
        for src in sources:
            print(f"  {src.file_path} "
                  f"({src.first_character_index}:{src.last_character_index})")

    def answer_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/answers",
                       skip_generation: bool = False) -> None:
        """Full RAG pipeline over an entire dataset: retrieve + generate.

        Loads Retriever and Generator once, then processes every question
        in a single pass. Use skip_generation=True to run retrieval only
        (useful for measuring recall@k without waiting for the LLM).

        Args:
            dataset_path: Path to a RagDataset JSON (answered or unanswered).
            k: Chunks to retrieve per question.
            save_directory: Where the output JSON is written.
            skip_generation: If True, saves SearchResults (no answers).
                             If False, saves SearchResultsAndAnswer.
        """
        if not os.path.isfile(dataset_path):
            print(f"Error: dataset file not found: {dataset_path}")
            return
        try:
            with open(dataset_path, "r", encoding="utf-8") as fh:
                dataset = RagDataset.model_validate_json(fh.read())
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return
        try:
            retriever = self._get_retriever()
        except Exception:
            return

        generator = None
        if not skip_generation:
            try:
                generator = self._get_generator()
            except Exception:
                return

        answers: list[MinimalAnswer] = []
        search_only: list[MinimalSearchResults] = []
        desc = "Retrieving" if skip_generation else "RAG pipeline"

        for question in tqdm(dataset.rag_questions, desc=desc):
            sources = retriever.search(question.question, k=k)

            if skip_generation or generator is None:
                search_only.append(MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                ))
            else:
                try:
                    response = generator.answer(question.question, sources)
                except Exception:
                    response = "Error generating answer."
                answers.append(MinimalAnswer(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                    answer=response,
                ))

        filename = os.path.basename(dataset_path)
        if skip_generation:
            self._save_json(
                StudentSearchResults(search_results=search_only, k=k),
                save_directory,
                filename,
                label="search_results",
            )
        else:
            self._save_json(
                StudentSearchResultsAndAnswer(search_results=answers, k=k),
                save_directory,
                filename,
                label="search_results_and_answer",
            )

    def evaluate(self, student_answer_path: str, dataset_path: str,
                 k: int = 10, max_context_length: int = 2000
                 ) -> None:
        """Evaluate retrieval quality using Recall@k vs ground-truth sources.

        Args:
            student_answer_path: Path to SearchResults JSON.
            dataset_path: Path to the AnsweredQuestions RagDataset JSON.
            k: Evaluate Recall@1, @3, @5 and @k.
            max_context_length: Kept for moulinette CLI compatibility.
        """
        from src.evaluator import Evaluator

        if not os.path.isfile(student_answer_path):
            print(f"Error: file not found: {student_answer_path}")
            return
        if not os.path.isfile(dataset_path):
            print(f"Error: ground truth not found: {dataset_path}")
            return
        try:
            evaluator = Evaluator()
            metrics = evaluator.compute_recall(
                student_path=student_answer_path,
                ground_truth_path=dataset_path,
                k=k,
            )
            evaluator.print_report(metrics)
        except Exception as e:
            print(f"Error during evaluation: {e}")

    @staticmethod
    def _save_json(model: object, directory: str,
                   filename: str, label: str) -> None:
        """Serialize a Pydantic model to JSON and write it to disk.

        Args:
            model: Any Pydantic BaseModel with model_dump_json().
            directory: Target directory (created if missing).
            filename: Output filename.
            label: Human-readable label for the success message.
        """
        os.makedirs(directory, exist_ok=True)
        out_path = os.path.join(directory, filename)
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(model.model_dump_json(indent=2))  # type: ignore[attr-defined]
            print(f"Saved {label} to {out_path}")
        except Exception as e:
            print(f"Error saving {label}: {e}")
