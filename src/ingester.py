import ast
import json
import os
import bm25s
from dataclasses import dataclass, asdict
from typing import Generator
from tqdm import tqdm


@dataclass
class Chunk:
    """A contiguous slice of a source file used as retrieval unit.

    Attributes:
        text: Raw text content of the chunk.
        file_path: Path to the source file, relative to the repo root.
        first_character_index: Inclusive start offset in the original file.
        last_character_index: Exclusive end offset in the original file.
        chunk_type: One of 'python', 'markdown', 'text'.
    """

    text: str
    file_path: str
    first_character_index: int
    last_character_index: int
    chunk_type: str

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Deserialize from a plain dict."""
        return cls(**data)


INCLUDE_EXTENSIONS = {".py", ".md", ".rst"}

EXCLUDE_DIRS = {
    "csrc", ".buildkite", ".github", "cmake",
    "docker", ".gemini", "requirements", "__pycache__",
    ".git", "node_modules",
}


def _should_index(path: str) -> bool:
    """Return True if this file should be included in the knowledge base.

    Args:
        path: Relative file path.

    Returns:
        True when extension is in INCLUDE_EXTENSIONS and no parent dir
        is in EXCLUDE_DIRS.
    """
    _, ext = os.path.splitext(path)
    if ext not in INCLUDE_EXTENSIONS:
        return False
    parts = set(path.replace("\\", "/").split("/"))
    return not parts.intersection(EXCLUDE_DIRS)


def _walk_repo(repo_root: str) -> Generator[str, None, None]:
    """Yield absolute paths of all indexable files under repo_root.

    Args:
        repo_root: Root directory of the vLLM repository.

    Yields:
        Absolute file paths that pass the _should_index filter.
    """
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS and not d.startswith(".")
        ]
        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, repo_root)
            if _should_index(rel_path):
                yield abs_path


def _split_by_size(
    text: str,
    file_path: str,
    chunk_type: str,
    max_chunk_size: int,
    start_offset: int = 0,
) -> list[Chunk]:
    """Split a text block into chunks of at most max_chunk_size chars.

    Breaks on newlines where possible to avoid cutting mid-line.

    Args:
        text: The text to split.
        file_path: Source file path for metadata.
        chunk_type: Label for the chunk type.
        max_chunk_size: Maximum characters per chunk.
        start_offset: Character offset of text[0] in the original file.

    Returns:
        List of Chunk objects.
    """
    chunks: list[Chunk] = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chunk_size, len(text))
        if end < len(text):
            newline = text.rfind("\n", pos, end)
            if newline > pos:
                end = newline + 1
        slice_text = text[pos:end].strip()
        if slice_text:
            chunks.append(Chunk(
                text=slice_text,
                file_path=file_path,
                first_character_index=start_offset + pos,
                last_character_index=start_offset + end,
                chunk_type=chunk_type,
            ))
        pos = end
    return chunks


def chunk_python_file(
    file_path: str,
    content: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Chunk a Python file using the AST to keep logical units intact.

    Extracts top-level functions and classes as individual chunks.
    Falls back to size-based splitting for oversized units or unparseable files.
    Module-level code (imports, constants) becomes a preamble chunk.

    Args:
        file_path: Relative path used for chunk metadata.
        content: Full text content of the Python file.
        max_chunk_size: Maximum characters per chunk.

    Returns:
        List of Chunk objects.
    """
    chunks: list[Chunk] = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _split_by_size(content, file_path, "python", max_chunk_size)

    lines = content.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)
    line_offsets.append(offset)

    def node_char_range(node: ast.AST) -> tuple[int, int]:
        start_line = getattr(node, "lineno", 1) - 1
        end_line = getattr(node, "end_lineno", start_line)
        first = line_offsets[start_line]
        last = (
            line_offsets[end_line + 1]
            if end_line + 1 < len(line_offsets)
            else len(content)
        )
        return first, last

    covered: list[tuple[int, int]] = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue

        first, last = node_char_range(node)
        node_text = content[first:last]
        covered.append((first, last))

        if len(node_text) <= max_chunk_size:
            chunks.append(Chunk(
                text=node_text.strip(),
                file_path=file_path,
                first_character_index=first,
                last_character_index=last,
                chunk_type="python",
            ))
        else:
            chunks.extend(
                _split_by_size(node_text, file_path, "python", max_chunk_size, first)
            )

    covered_sorted = sorted(covered)
    preamble_parts: list[str] = []
    preamble_start = 0

    for seg_start, seg_end in covered_sorted:
        gap = content[preamble_start:seg_start].strip()
        if gap:
            preamble_parts.append(gap)
        preamble_start = seg_end

    tail = content[preamble_start:].strip()
    if tail:
        preamble_parts.append(tail)

    preamble = "\n\n".join(preamble_parts).strip()
    if preamble:
        chunks.extend(
            _split_by_size(preamble, file_path, "python", max_chunk_size)
        )

    return chunks


def chunk_markdown_file(
    file_path: str,
    content: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Chunk a Markdown or RST file by splitting on header lines.

    Each section (header to next header) becomes one chunk.
    Oversized sections are split further by _split_by_size.

    Args:
        file_path: Relative path used for chunk metadata.
        content: Full text content of the file.
        max_chunk_size: Maximum characters per chunk.

    Returns:
        List of Chunk objects.
    """
    chunks: list[Chunk] = []
    lines = content.splitlines(keepends=True)

    section_lines: list[str] = []
    section_start_offset = 0
    char_offset = 0

    for line in lines:
        is_header = line.startswith("#")
        if is_header and section_lines:
            section_text = "".join(section_lines).strip()
            if section_text:
                if len(section_text) <= max_chunk_size:
                    chunks.append(Chunk(
                        text=section_text,
                        file_path=file_path,
                        first_character_index=section_start_offset,
                        last_character_index=char_offset,
                        chunk_type="markdown",
                    ))
                else:
                    chunks.extend(_split_by_size(
                        section_text, file_path, "markdown",
                        max_chunk_size, section_start_offset,
                    ))
            section_lines = []
            section_start_offset = char_offset

        section_lines.append(line)
        char_offset += len(line)

    if section_lines:
        section_text = "".join(section_lines).strip()
        if section_text:
            if len(section_text) <= max_chunk_size:
                chunks.append(Chunk(
                    text=section_text,
                    file_path=file_path,
                    first_character_index=section_start_offset,
                    last_character_index=char_offset,
                    chunk_type="markdown",
                ))
            else:
                chunks.extend(_split_by_size(
                    section_text, file_path, "markdown",
                    max_chunk_size, section_start_offset,
                ))

    return chunks


class Ingester:
    """Reads the vLLM repository, chunks it, and builds a BM25 index.

    Usage:
        ingester = Ingester(repo_root="data/raw/vllm-0.10.1", max_chunk_size=2000)
        ingester.build()
        ingester.save("data/processed")

        # Reload without re-indexing:
        ingester = Ingester.load("data/processed")

    Attributes:
        repo_root: Path to the root of the vLLM repository.
        max_chunk_size: Maximum characters per chunk.
        chunks: All Chunk objects produced by build().
        retriever: Fitted bm25s.BM25 instance produced by build().
    """

    CHUNKS_FILE = "chunks.json"
    INDEX_DIR = "bm25_index"

    def __init__(self, repo_root: str, max_chunk_size: int = 2000) -> None:
        """Initialize the Ingester.

        Args:
            repo_root: Path to the vLLM repository root directory.
            max_chunk_size: Maximum number of characters per chunk.
        """
        self.repo_root = repo_root
        self.max_chunk_size = max_chunk_size
        self.chunks: list[Chunk] = []
        self.retriever: bm25s.BM25 | None = None

    def build(self) -> None:
        """Ingest the repository: chunk all files and fit the BM25 index."""
        self.chunks = self._collect_chunks()
        self.retriever = self._build_bm25(self.chunks)

    def save(self, output_dir: str) -> None:
        """Persist chunks and BM25 index to disk.

        Args:
            output_dir: Directory where artefacts will be written.

        Raises:
            RuntimeError: If build() has not been called yet.
        """
        if not self.chunks or self.retriever is None:
            raise RuntimeError("Call build() before save().")

        chunks_dir = os.path.join(output_dir, "chunks")
        index_dir = os.path.join(output_dir, self.INDEX_DIR)
        os.makedirs(chunks_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)

        chunks_path = os.path.join(chunks_dir, self.CHUNKS_FILE)
        with open(chunks_path, "w", encoding="utf-8") as fh:
            json.dump([c.to_dict() for c in self.chunks], fh, ensure_ascii=False)

        self.retriever.save(index_dir)
        print(f"Saved {len(self.chunks)} chunks → {chunks_dir}")
        print(f"Saved BM25 index → {index_dir}")

    @classmethod
    def load(cls, processed_dir: str) -> "Ingester":
        """Load a previously built Ingester from disk.

        Args:
            processed_dir: Directory produced by save().

        Returns:
            Ingester with chunks and retriever populated.
        """
        ingester = cls.__new__(cls)
        ingester.repo_root = ""
        ingester.max_chunk_size = 2000

        chunks_path = os.path.join(processed_dir, "chunks", cls.CHUNKS_FILE)
        with open(chunks_path, "r", encoding="utf-8") as fh:
            ingester.chunks = [Chunk.from_dict(d) for d in json.load(fh)]

        index_dir = os.path.join(processed_dir, cls.INDEX_DIR)
        ingester.retriever = bm25s.BM25.load(index_dir, load_corpus=False)

        print(f"Loaded {len(ingester.chunks)} chunks from {processed_dir}")
        return ingester

    def _collect_chunks(self) -> list[Chunk]:
        """Walk the repository and produce all chunks."""
        all_chunks: list[Chunk] = []
        files = list(_walk_repo(self.repo_root))

        for abs_path in tqdm(files, desc="Chunking files"):
            rel_path = os.path.relpath(abs_path)  # relativo al CWD del proyecto
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except OSError:
                continue

            if not content.strip():
                continue

            _, ext = os.path.splitext(abs_path)
            if ext == ".py":
                file_chunks = chunk_python_file(
                    rel_path, content, self.max_chunk_size
                )
            else:
                file_chunks = chunk_markdown_file(
                    rel_path, content, self.max_chunk_size
                )

            all_chunks.extend(file_chunks)

        return all_chunks

    def _build_bm25(self, chunks: list[Chunk]) -> bm25s.BM25:
        """Tokenize chunks and fit the BM25 model.

        Args:
            chunks: List of Chunk objects to index.

        Returns:
            A fitted bm25s.BM25 instance.
        """
        corpus = [c.text for c in chunks]
        print("Tokenizing corpus...")
        tokenized = bm25s.tokenize(corpus, stopwords="en", show_progress=True)
        print("Fitting BM25...")
        retriever = bm25s.BM25()
        retriever.index(tokenized, show_progress=True)
        return retriever
