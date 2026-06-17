from src.retriever import Retriever
from src.generator import Generator
import os
from typing import Optional
from tqdm import tqdm
from src.models import (MinimalAnswer,
                        MinimalSearchResults,
                        RagDataset,
                        StudentSearchResults,
                        StudentSearchResultsAndAnswer,
                        MinimalSource)

PROCESSED_DIR = "data/processed"
REPO_ROOT = "data/raw/vllm-0.10.1"


class CLI:
    """RAG against the Machine — single orchestrator and CLI entry point.

    Loads Retriever and Generator lazily on first use and reuses them
    for all subsequent calls within the same process.

    Two retrieval modes are available:
        search / search_dataset  — fast BM25-only, no reranker.
        answer / answer_dataset  — BM25 + reranker before LLM generation.

    Commands:
        index           Build BM25 and optional embedding indices.
        search          Search for a single query (fast, no reranker).
        search_dataset  Batch retrieval over a full dataset JSON.
        answer          Answer a single question (with reranker).
        answer_dataset  Full pipeline: retrieve + rerank + generate.
        evaluate        Compute Recall@k against ground truth.

    Examples:
        uv run python -m src index
        uv run python -m src search "How to configure OpenAI server?" --k 10
        uv run python -m src answer "How to configure OpenAI server?" --k 10
        uv run python -m src search_dataset \\
            --dataset_path

            data/datasets/UnansweredQuestions/dataset_docs_public.json
        uv run python -m src answer_dataset \\
            --dataset_path

            data/datasets/UnansweredQuestions/dataset_docs_public.json
        uv run python -m src evaluate \\
            --student_answer_path

            data/output/search_results/dataset_docs_public.json \\
            --dataset_path

            data/datasets/AnsweredQuestions/dataset_docs_public.json
    """

    def __init__(self, processed_dir: str = PROCESSED_DIR,
                 repo_root: str = REPO_ROOT) -> None:
        """Initialise the CLI orchestrator.

        Args:
            processed_dir: Directory containing chunks and BM25 index.
            repo_root: Root of the vLLM repository (used by Generator).
        """
        self.processed_dir = processed_dir
        self.repo_root = repo_root
        self.retriever: Optional[object] = None
        self.generator: Optional[object] = None

    def get_retriever(self, with_reranker: bool = False) -> Retriever:
        """Load and cache the Retriever.

        Args:
            with_reranker: When True, also loads the cross-encoder reranker.

        Returns:
            The loaded Retriever instance.

        Raises:
            Exception: Propagates any load error after printing a message.
        """
        if self.retriever is None:
            try:
                self.retriever = Retriever.from_disk(self.processed_dir)
            except Exception as exc:
                print(f"Error loading index: {exc}")
                print("Have you run 'uv run python -m src index' yet?")
                raise

        if with_reranker and self.retriever.reranker is None:
            self.retriever.load_reranker()

        return self.retriever

    def get_generator(self) -> Generator:
        """Load and cache the Generator.

        Returns:
            The loaded Generator instance.

        Raises:
            Exception: Propagates any model load error.
        """
        if self.generator is None:
            try:
                self.generator = Generator(repo_root=self.repo_root)
            except Exception as exc:
                print(f"Error loading model: {exc}")
                raise
        return self.generator

    def index(self, repo_root: str = REPO_ROOT, code_chunk_size: int = 1200,
              doc_chunk_size: int = 2000, output_dir: str = PROCESSED_DIR,
              use_embeddings: bool = True) -> None:
        """Ingest the vLLM repository and persist BM25 and embedding indices.

        Args:
            repo_root: Path to the vLLM repository root.
            code_chunk_size: Maximum characters per code chunk.
            doc_chunk_size: Maximum characters per doc chunk.
            output_dir: Where chunks and indices are saved.
            use_embeddings: Build semantic embedding index alongside BM25.
        """
        from src.ingester import Ingester

        if not os.path.isdir(repo_root):
            print(f"Error: repository not found at '{repo_root}'")
            return
        try:
            ingester = Ingester(
                repo_root=repo_root,
                code_chunk_size=code_chunk_size,
                doc_chunk_size=doc_chunk_size,
            )
            ingester.build(use_embeddings=use_embeddings)
            ingester.save(output_dir)
            print("Ingestion complete! Indices saved under", output_dir)
        except Exception as exc:
            print(f"Error during indexing: {exc}")

    def search(self, query: str, k: int = 10) -> None:
        """Search the knowledge base for a single query and print results.

        Uses fast BM25-only retrieval — no reranker.

        Args:
            query: Search string. Empty query exits cleanly.
            k: Number of results to retrieve.
        """
        if not query or not query.strip():
            print("Empty query — no results.")
            return
        try:
            retriever = self.get_retriever(with_reranker=False)
            results = retriever.search(query, k=k)
        except Exception as exc:
            print(f"Error during search: {exc}")
            return

        if not results:
            print("No results found.")
            return

        print(f"\nPrompt: \"{query}\"\n")
        for i, source in enumerate(results, 1):
            chunk = self.find_chunk(retriever, source)
            print(f"[{i}] filepath: {source.file_path}")
            print(f"    chunk:    {i - 1}")
            print(f"    range:    {source.first_character_index}:"
                  f"{source.last_character_index}")
            if chunk:
                preview = chunk.text[:200].replace("\n", " ")
                print(f"    text:     {preview}...")
            print()

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/search_results"
                       ) -> None:
        """Run fast retrieval over every question in a dataset JSON.

        Uses BM25-only search without reranker for evaluation throughput.

        Args:
            dataset_path: Path to a RagDataset JSON.
            k: Chunks to retrieve per question.
            save_directory: Where the StudentSearchResults JSON is
            written.
        """
        if not os.path.isfile(dataset_path):
            print(f"Error: dataset file not found: {dataset_path}")
            return
        try:
            with open(dataset_path, "r", encoding="utf-8") as fh:
                dataset = RagDataset.model_validate_json(fh.read())
        except Exception as exc:
            print(f"Error reading dataset: {exc}")
            return
        try:
            retriever = self.get_retriever(with_reranker=False)
        except Exception:
            return

        results: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            sources = retriever.search_with_fallback(question.question, k=k)
            results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources,
            ))

        self.save_json(
            StudentSearchResults(search_results=results, k=k),
            save_directory,
            os.path.basename(dataset_path),
            label="student_search_results",
        )

    def answer(self, query: str, k: int = 10) -> None:
        """Answer a single question using retrieved and reranked context.

        Args:
            query: Natural-language question.
            k: Number of chunks to retrieve as context.
        """
        if not query or not query.strip():
            print("Empty query — nothing to answer.")
            return
        try:
            retriever = self.get_retriever(with_reranker=True)
            sources = retriever.search_for_generation(query, k=k)
        except Exception as exc:
            print(f"Error during retrieval: {exc}")
            return

        if not sources:
            print("No relevant sources found.")
            return

        try:
            generator = self.get_generator()
            response = generator.answer(query, sources)
        except Exception as exc:
            print(f"Error during generation: {exc}")
            return

        print(f"\nPrompt: \"{query}\"\n")
        print("=== Answer ===")
        print(response)

        not_found = "Not found in the provided sources" in response
        if not not_found:
            print("\n=== Sources ===")
            for i, src in enumerate(sources):
                chunk = self.find_chunk(retriever, src)
                print(f"[{i + 1}] filepath: {src.file_path}")
                print(f"    chunk:    {i}")
                print(f"    range:    {src.first_character_index}:"
                      f"{src.last_character_index}")
                if chunk:
                    preview = chunk.text[:200].replace("\n", " ")
                    print(f"    text:     {preview}...")
                print()

    def answer_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/answers",
                       skip_generation: bool = False) -> None:
        """Full RAG pipeline over an entire dataset:
        retrieve + rerank + generate.

        Args:
            dataset_path: Path to a RagDataset JSON.
            k: Chunks to retrieve per question.
            save_directory: Where the output JSON is written.
            skip_generation: If True, saves retrieval results only.
                Useful for measuring recall@k without the LLM.
        """
        if not os.path.isfile(dataset_path):
            print(f"Error: dataset file not found: {dataset_path}")
            return
        try:
            with open(dataset_path, "r", encoding="utf-8") as fh:
                dataset = RagDataset.model_validate_json(fh.read())
        except Exception as exc:
            print(f"Error reading dataset: {exc}")
            return

        use_reranker = not skip_generation
        try:
            retriever = self.get_retriever(with_reranker=use_reranker)
        except Exception:
            return

        generator = None
        if not skip_generation:
            try:
                generator = self.get_generator()
            except Exception:
                return

        answers: list[MinimalAnswer] = []
        search_only: list[MinimalSearchResults] = []
        desc = "Retrieving" if skip_generation else "RAG pipeline"

        for question in tqdm(dataset.rag_questions, desc=desc):
            if skip_generation:
                sources = retriever.search(question.question, k=k)
                search_only.append(MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                ))
            else:
                sources = retriever.search_for_generation(
                    question.question, k=k
                )
                try:
                    response = generator.answer(  # type: ignore
                        question.question, sources
                    )
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
            self.save_json(
                StudentSearchResults(search_results=search_only, k=k),
                save_directory,
                filename,
                label="search_results",
            )
        else:
            self.save_json(
                StudentSearchResultsAndAnswer(
                    search_results=answers, k=k
                ),
                save_directory,
                filename,
                label="search_results_and_answer",
            )

    def evaluate(self, student_answer_path: str, dataset_path: str,
                 k: int = 10, max_context_length: int = 2000) -> None:
        """Evaluate retrieval quality using Recall@k vs ground-truth sources.

        Args:
            student_answer_path: Path to StudentSearchResults JSON.
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
                k=k)
            evaluator.print_report(metrics)
        except Exception as exc:
            print(f"Error during evaluation: {exc}")

    @staticmethod
    def find_chunk(retriever: object,
                   source: MinimalSource) -> Optional[object]:
        """Find the Chunk object matching a MinimalSource.

        Args:
            retriever: The loaded Retriever instance.
            source: MinimalSource to look up.

        Returns:
            Matching Chunk or None if not found.
        """
        for chunk in retriever.chunks:
            if (
                chunk.file_path == source.file_path
                and chunk.first_character_index
                == source.first_character_index
            ):
                return chunk
        return None

    @staticmethod
    def save_json(model: object, directory: str,
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
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(model.model_dump_json(indent=2))
            print(f"Saved {label} to {out_path}")
        except Exception as exc:
            print(f"Error saving {label}: {exc}")
