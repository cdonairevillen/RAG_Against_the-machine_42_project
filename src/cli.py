from src.retriever import Retriever
from src.generator import Generator
from src.ingester import Chunk
from pydantic import BaseModel
import os
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
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
        uv run python -m src answer_dataset \\
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
        uv run python -m src evaluate \\
    --student_answer_path data/output/search_results/dataset_docs_public.json
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
    """

    def __init__(self, processed_dir: str = PROCESSED_DIR,
                 repo_root: str = REPO_ROOT) -> None:
        self.processed_dir = processed_dir
        self.repo_root = repo_root
        self.retriever: Retriever | None = None
        self.generator: Generator | None = None

    def get_retriever(self, with_reranker: bool = True) -> Retriever:
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
        if self.generator is None:
            try:
                self.generator = Generator(repo_root=self.repo_root)
            except Exception as exc:
                print(f"Error loading model: {exc}")
                raise
        return self.generator

    def index(self, repo_root: str = REPO_ROOT,
              max_chunk_size: int = 2000,
              code_chunk_size: int = 2000, doc_chunk_size: int = 2000,
              output_dir: str = PROCESSED_DIR,
              use_embeddings: bool = False) -> None:
        from src.ingester import Ingester

        if max_chunk_size != 2000:
            code_chunk_size = max_chunk_size
            doc_chunk_size = max_chunk_size

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
        """Run dual BM25 retrieval over every question in a dataset JSON.

        Uses separate doc and code indices interleaved — no reranker.

        Args:
            dataset_path: Path to a RagDataset JSON.
            k: Chunks to retrieve per question.
            save_directory: Where the StudentSearchResults JSON is written.
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
            retriever = self.get_retriever(with_reranker=True)
        except Exception:
            return

        results: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            sources = retriever.search_smart(question.question, k=k)
            if sources is None:
                sources = []
            results.append(MinimalSearchResults(
                question_id=question.question_id,
                question_str=question.question,
                retrieved_sources=sources,
            ))

        self.save_json(
            StudentSearchResults(search_results=results, k=k),
            save_directory,
            os.path.basename(dataset_path),
            label="student_search_results",
        )

    def answer(self, query: str, k: int = 10) -> None:
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

        parent_map = retriever.get_parent_chunk_map()
        context_blocks = []
        for src in sources:
            parent = parent_map.get(
                (src.file_path, src.first_character_index)
            )
            if parent:
                context_blocks.append(
                    f"--- file: {src.file_path} ---\n{parent.text}"
                )
        try:
            generator = self.get_generator()
            response = generator.answer_with_text(query, context_blocks)
        except Exception as exc:
            print(f"Error during generation: {exc}")
            return

        print(f"\nPrompt: \"{query}\"\n")
        print("=== Answer ===")
        print(response)

        not_found_phrases = [
            "not found in the provided sources",
            "provided sources do not contain",
            "no relevant information",
            "i've not found",
        ]
        not_found = any(p in response.lower() for p in not_found_phrases)
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
                       skip_generation: bool = True) -> None:
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
            retriever = self.get_retriever(with_reranker=True)
        except Exception:
            return

        if not skip_generation:
            try:
                self.generator = self.get_generator()
            except Exception:
                return

        answers: list[MinimalAnswer] = []
        search_only: list[MinimalSearchResults] = []
        desc = "Retrieving" if skip_generation else "RAG pipeline"

        if skip_generation:
            for question in tqdm(dataset.rag_questions, desc=desc):
                sources = retriever.search_for_generation(
                    question.question, k=k
                )
                if sources is None:
                    sources = []
                search_only.append(MinimalSearchResults(
                    question_id=question.question_id,
                    question_str=question.question,
                    retrieved_sources=sources,
                ))
        else:
            parent_map = retriever.get_parent_chunk_map()
            for question in tqdm(dataset.rag_questions, desc=desc):
                sources = retriever.search_for_generation(
                    question.question, k=k
                )
                if sources is None:
                    sources = []
                context_blocks = []
                for src in sources:
                    parent = parent_map.get(
                        (src.file_path, src.first_character_index)
                    )
                    if parent:
                        context_blocks.append(
                            f"--- file: {src.file_path} ---\n{parent.text}"
                        )
                try:
                    response = self.generator.answer_with_text(
                        question.question, context_blocks
                    )
                except Exception:
                    response = "Error generating answer."
                answers.append(MinimalAnswer(
                    question_id=question.question_id,
                    question_str=question.question,
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
    def find_chunk(retriever: Retriever,
                   source: MinimalSource) -> Chunk | None:
        chunk_map = retriever.get_chunk_map()
        return chunk_map.get(
            (source.file_path, source.first_character_index))

    @staticmethod
    def save_json(model: BaseModel, directory: str,
                  filename: str, label: str) -> None:
        os.makedirs(directory, exist_ok=True)
        out_path = os.path.join(directory, filename)
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(model.model_dump_json(indent=2))
            print(f"Saved {label} to {out_path}")
        except Exception as exc:
            print(f"Error saving {label}: {exc}")
