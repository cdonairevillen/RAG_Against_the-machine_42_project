import ast
import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from typing import Generator, Optional
import bm25s
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


INCLUDE_EXTENSIONS = {".py", ".pyi", ".md", ".rst", ".txt"}

EXCLUDE_DIRS = {
    "csrc", ".buildkite", ".github", "cmake",
    "docker", ".gemini", "requirements", "__pycache__",
    ".git", "node_modules",
}

CODE_EXTENSIONS = {".py", ".pyi"}
DOC_EXTENSIONS = {".md", ".rst", ".txt"}

OVERLAP_RATIO = 0.20
DEFAULT_CODE_CHUNK_SIZE = 1200
DEFAULT_DOC_CHUNK_SIZE = 2000


@dataclass
class Chunk:
    """A contiguous slice of a source file used as retrieval unit.

    Attributes:
        text: Raw text content of the chunk (used for BM25 indexing).
        file_path: Path relative to the project working directory.
        first_character_index: Inclusive start offset in the original file.
        last_character_index: Exclusive end offset in the original file.
        chunk_type: One of 'code' or 'doc'.
        symbols: Space-separated AST identifier names extracted from the
            chunk. Stored separately so they do not affect BM25 length
            normalization but can be used for query expansion.
    """

    text: str
    file_path: str
    first_character_index: int
    last_character_index: int
    chunk_type: str
    symbols: str = field(default="")

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Deserialize from a plain dict."""
        return cls(**data)


def walk_repo(repo_root: str) -> Generator[str, None, None]:
    """Yield absolute paths of all indexable files under repo_root.

    Args:
        repo_root: Root directory of the vLLM repository.

    Yields:
        Absolute file paths passing extension and directory filters.
    """
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS and not d.startswith(".")
        ]
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext not in INCLUDE_EXTENSIONS:
                continue
            abs_path = os.path.join(dirpath, filename)
            rel_parts = set(
                os.path.relpath(abs_path, repo_root)
                .replace("\\", "/")
                .split("/")
            )
            if not rel_parts.intersection(EXCLUDE_DIRS):
                yield abs_path


def split_by_size(text: str, file_path: str, chunk_type: str,
                  max_chunk_size: int, start_offset: int = 0,
                  symbols: str = "") -> list[Chunk]:
    """Split text into overlapping chunks of at most max_chunk_size chars.

    Each chunk overlaps the previous by OVERLAP_RATIO * max_chunk_size
    characters so that context at chunk boundaries is not lost.

    Args:
        text: The text to split.
        file_path: Source file path for metadata.
        chunk_type: Label for the chunk type ('code' or 'doc').
        max_chunk_size: Maximum characters per chunk.
        start_offset: Character offset of text[0] in the original file.
        symbols: AST symbol names to store on each produced chunk.

    Returns:
        List of overlapping Chunk objects.
    """
    chunks: list[Chunk] = []
    overlap = int(max_chunk_size * OVERLAP_RATIO)
    step = max_chunk_size - overlap
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
                symbols=symbols,
            ))
        pos += step

    return chunks


def extract_symbols(node: ast.AST) -> str:
    """Extract identifier names from an AST node as a space-separated string.

    Collects class names, method names and attribute names defined within
    the node. These are stored in Chunk.symbols separately from the BM25
    corpus text so they do not inflate chunk length.

    Args:
        node: Top-level AST node (FunctionDef, AsyncFunctionDef, ClassDef).

    Returns:
        Space-separated symbol names, or empty string if none found.
    """
    names: list[str] = []

    if isinstance(node, ast.ClassDef):
        names.append(node.name)
        for child in ast.walk(node):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                names.append(child.name)
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        names.append(target.id)
            if isinstance(child, ast.AnnAssign):
                if isinstance(child.target, ast.Name):
                    names.append(child.target.id)

    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        names.append(node.name)
        for arg in node.args.args:
            names.append(arg.arg)

    if not names:
        return ""

    return " ".join(dict.fromkeys(names))


def chunk_python_file(file_path: str, content: str,
                      max_chunk_size: int = DEFAULT_CODE_CHUNK_SIZE
                      ) -> list[Chunk]:
    """Chunk a Python file using the AST to keep logical units intact.

    AST symbol names are stored in Chunk.symbols, not in Chunk.text,
    so they do not affect BM25 length normalization. The Retriever uses
    them for query expansion at search time.

    Args:
        file_path: Relative path used for chunk metadata.
        content: Full text content of the Python file.
        max_chunk_size: Maximum characters per chunk.

    Returns:
        List of Chunk objects with symbols metadata.
    """
    chunks: list[Chunk] = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return split_by_size(content, file_path, "code", max_chunk_size)

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
        symbols = extract_symbols(node)
        covered.append((first, last))

        if len(node_text) <= max_chunk_size:
            chunks.append(Chunk(
                text=node_text.strip(),
                file_path=file_path,
                first_character_index=first,
                last_character_index=last,
                chunk_type="code",
                symbols=symbols,
            ))
        else:
            chunks.extend(
                split_by_size(
                    node_text, file_path, "code",
                    max_chunk_size, first, symbols,
                )
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
            split_by_size(preamble, file_path, "code", max_chunk_size)
        )

    return chunks


def chunk_doc_file(file_path: str, content: str,
                   max_chunk_size: int = DEFAULT_DOC_CHUNK_SIZE
                   ) -> list[Chunk]:
    """Chunk a Markdown or RST file by splitting on header lines.

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

    def flush(end_offset: int) -> None:
        section_text = "".join(section_lines).strip()
        if not section_text:
            return
        if len(section_text) <= max_chunk_size:
            chunks.append(Chunk(
                text=section_text,
                file_path=file_path,
                first_character_index=section_start_offset,
                last_character_index=end_offset,
                chunk_type="doc",
            ))
        else:
            chunks.extend(split_by_size(
                section_text, file_path, "doc",
                max_chunk_size, section_start_offset,
            ))

    for line in lines:
        if line.startswith("#") and section_lines:
            flush(char_offset)
            section_lines = []
            section_start_offset = char_offset
        section_lines.append(line)
        char_offset += len(line)

    flush(char_offset)
    return chunks


def chunk_text_file(file_path: str, content: str,
                    max_chunk_size: int = DEFAULT_DOC_CHUNK_SIZE
                    ) -> list[Chunk]:
    """Fallback chunker for plain text and unrecognised readable file types.

    Args:
        file_path: Relative path used for chunk metadata.
        content: Full text content of the file.
        max_chunk_size: Maximum characters per chunk.

    Returns:
        List of Chunk objects with overlap.
    """
    return split_by_size(content, file_path, "doc", max_chunk_size)


class Ingester:
    """Reads the vLLM repository, chunks it, and builds a unified BM25 index.

    Uses different chunk sizes for code (1200) and docs (2000) to improve
    term density in code chunks without sacrificing doc context coverage.
    AST symbol names are stored in Chunk.symbols separately from the BM25
    corpus to avoid penalising chunks via length normalization.

    Usage:
        ingester = Ingester(repo_root="data/raw/vllm-0.10.1")
        ingester.build()
        ingester.save("data/processed")

        ingester = Ingester.load("data/processed")

    Attributes:
        repo_root: Path to the root of the vLLM repository.
        code_chunk_size: Maximum characters per code chunk.
        doc_chunk_size: Maximum characters per doc chunk.
        chunks: All Chunk objects produced by build().
        bm25: Fitted unified BM25 instance.
        embeddings: Dense vectors for all chunks, or None.
    """

    CHUNKS_FILE = "chunks.json"
    INDEX_DIR = "bm25_index"
    EMBEDDINGS_FILE = "embeddings.npy"
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, repo_root: str,
                 code_chunk_size: int = DEFAULT_CODE_CHUNK_SIZE,
                 doc_chunk_size: int = DEFAULT_DOC_CHUNK_SIZE) -> None:
        """Initialise the Ingester.

        Args:
            repo_root: Path to the vLLM repository root directory.
            code_chunk_size: Maximum characters per code chunk.
            doc_chunk_size: Maximum characters per doc chunk.
        """
        self.repo_root = repo_root
        self.code_chunk_size = code_chunk_size
        self.doc_chunk_size = doc_chunk_size
        self.chunks: list[Chunk] = []
        self.bm25: Optional[bm25s.BM25] = None
        self.bm25_docs: Optional[bm25s.BM25] = None
        self.bm25_code: Optional[bm25s.BM25] = None
        self.doc_indices: list[int] = []
        self.code_indices: list[int] = []
        self.embeddings: Optional[np.ndarray] = None

    def build(self, use_embeddings: bool = True) -> None:
        """Ingest the repository and fit the BM25 index.

        Args:
            use_embeddings: When True, also encodes all chunks with
                sentence-transformers for semantic search.
        """
        self.chunks = self.collect_chunks()
        self.bm25 = self.build_bm25(self.chunks)
        self.doc_indices = [
            i for i, c in enumerate(self.chunks) if c.chunk_type == "doc"
        ]
        self.code_indices = [
            i for i, c in enumerate(self.chunks) if c.chunk_type == "code"
        ]
        self.bm25_docs = self.build_bm25(
            [self.chunks[i] for i in self.doc_indices], label="docs"
        )
        self.bm25_code = self.build_bm25(
            [self.chunks[i] for i in self.code_indices], label="code"
        )
        if use_embeddings:
            self.embeddings = self.build_embeddings(self.chunks)

    def save(self, output_dir: str) -> None:
        """Persist all artefacts to disk.

        Args:
            output_dir: Root directory for all output artefacts.

        Raises:
            RuntimeError: If build() has not been called.
        """
        if not self.chunks or self.bm25 is None:
            raise RuntimeError("Call build() before save().")

        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        chunks_dir = os.path.join(output_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)

        with open(
            os.path.join(chunks_dir, self.CHUNKS_FILE), "w", encoding="utf-8"
        ) as fh:
            json.dump(
                [c.to_dict() for c in self.chunks], fh, ensure_ascii=False
            )

        index_dir = os.path.join(output_dir, self.INDEX_DIR)
        os.makedirs(index_dir, exist_ok=True)
        self.bm25.save(index_dir)
        docs_dir = os.path.join(output_dir, "bm25_docs")
        code_dir = os.path.join(output_dir, "bm25_code")
        os.makedirs(docs_dir, exist_ok=True)
        os.makedirs(code_dir, exist_ok=True)
        self.bm25_docs.save(docs_dir)
        self.bm25_code.save(code_dir)

        meta_path = os.path.join(output_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({
                "doc_indices": self.doc_indices,
                "code_indices": self.code_indices,
            }, fh)

        doc_count = sum(1 for c in self.chunks if c.chunk_type == "doc")
        code_count = sum(1 for c in self.chunks if c.chunk_type == "code")
        print(
            f"Saved {len(self.chunks)} chunks "
            f"({doc_count} doc, {code_count} code)"
        )

        if self.embeddings is not None:
            embed_path = os.path.join(output_dir, self.EMBEDDINGS_FILE)
            np.save(embed_path, self.embeddings)
            print(f"Saved embeddings {self.embeddings.shape} → {embed_path}")

    @classmethod
    def load(cls, processed_dir: str) -> "Ingester":
        """Load a previously built Ingester from disk.

        Args:
            processed_dir: Directory produced by save().

        Returns:
            Ingester instance ready for use by the Retriever.
        """
        ingester = cls.__new__(cls)
        ingester.repo_root = ""
        ingester.code_chunk_size = DEFAULT_CODE_CHUNK_SIZE
        ingester.doc_chunk_size = DEFAULT_DOC_CHUNK_SIZE

        chunks_dir = os.path.join(processed_dir, "chunks")
        with open(
            os.path.join(chunks_dir, cls.CHUNKS_FILE), "r", encoding="utf-8"
        ) as fh:
            ingester.chunks = [Chunk.from_dict(d) for d in json.load(fh)]

        index_dir = os.path.join(processed_dir, cls.INDEX_DIR)
        ingester.bm25 = bm25s.BM25.load(index_dir, load_corpus=False)

        docs_dir = os.path.join(processed_dir, "bm25_docs")
        code_dir = os.path.join(processed_dir, "bm25_code")
        meta_path = os.path.join(processed_dir, "meta.json")

        if os.path.isdir(docs_dir) and os.path.isdir(code_dir):
            ingester.bm25_docs = bm25s.BM25.load(docs_dir, load_corpus=False)
            ingester.bm25_code = bm25s.BM25.load(code_dir, load_corpus=False)
        else:
            ingester.bm25_docs = None
            ingester.bm25_code = None

        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            ingester.doc_indices = meta["doc_indices"]
            ingester.code_indices = meta["code_indices"]
        else:
            ingester.doc_indices = []
            ingester.code_indices = []

        embed_path = os.path.join(processed_dir, cls.EMBEDDINGS_FILE)
        if os.path.isfile(embed_path):
            ingester.embeddings = np.load(embed_path)
            print(f"Loaded embeddings {ingester.embeddings.shape}")
        else:
            ingester.embeddings = None

        doc_count = sum(1 for c in ingester.chunks if c.chunk_type == "doc")
        code_count = sum(
            1 for c in ingester.chunks if c.chunk_type == "code"
        )
        print(
            f"Loaded {len(ingester.chunks)} chunks "
            f"({doc_count} doc, {code_count} code)"
        )
        return ingester

    def collect_chunks(self) -> list[Chunk]:
        """Walk the repository and produce all chunks.

        Returns:
            Flat list of Chunk objects from all indexed files.
        """
        all_chunks: list[Chunk] = []
        files = list(walk_repo(self.repo_root))

        for abs_path in tqdm(files, desc="Chunking files"):
            rel_path = os.path.relpath(abs_path).replace("\\", "/")
            try:
                with open(
                    abs_path, "r", encoding="utf-8", errors="ignore"
                ) as fh:
                    content = fh.read()
            except OSError:
                continue

            if not content.strip():
                continue

            _, ext = os.path.splitext(abs_path)
            if ext in CODE_EXTENSIONS:
                file_chunks = chunk_python_file(
                    rel_path, content, self.code_chunk_size
                )
            elif ext in DOC_EXTENSIONS:
                file_chunks = chunk_doc_file(
                    rel_path, content, self.doc_chunk_size
                )
            else:
                file_chunks = chunk_text_file(
                    rel_path, content, self.doc_chunk_size
                )

            all_chunks.extend(file_chunks)

        return all_chunks

    def build_bm25(self, chunks: list[Chunk],
                   label: str = "corpus") -> bm25s.BM25:
        """Tokenize all chunks and fit a unified BM25 model.

        Only Chunk.text is used — Chunk.symbols is excluded from the
        corpus to avoid penalising chunks via BM25 length normalization.

        Args:
            chunks: All Chunk objects to index.
            label: Human-readable label for progress output.

        Returns:
            A fitted bm25s.BM25 instance.
        """
        corpus = [c.text for c in chunks]
        print(f"Tokenizing {label} ({len(corpus)} chunks)...")
        tokenized = bm25s.tokenize(
            corpus, stopwords="en", show_progress=True
        )
        retriever = bm25s.BM25()
        retriever.index(tokenized, show_progress=True)
        return retriever

    def build_embeddings(self, chunks: list[Chunk]) -> np.ndarray:
        """Encode all chunks into dense embedding vectors.

        Uses all-MiniLM-L6-v2 (80 MB, 384 dimensions).
        Vectors are L2-normalised so cosine similarity equals dot product.

        Args:
            chunks: All Chunk objects to encode.

        Returns:
            numpy array of shape (n_chunks, 384), dtype float32.
        """

        print(f"Loading embedding model {self.EMBED_MODEL}...")
        model = SentenceTransformer(self.EMBED_MODEL)
        texts = [c.text for c in chunks]
        print(f"Encoding {len(texts)} chunks...")
        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings
