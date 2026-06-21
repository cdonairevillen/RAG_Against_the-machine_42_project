import ast
import json
import os
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Generator, Optional
import bm25s
import numpy as np
from tqdm import tqdm


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

DEFAULT_CODE_CHILD_SIZE = 600
DEFAULT_DOC_CHILD_SIZE = 600


@dataclass
class Chunk():
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
    parent_id: int = field(default=-1)

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
        elif isinstance(node, ast.ClassDef):
            class_sig = node_text.split("\n")[0]
            method_covered: list[tuple[int, int]] = []

            for child in ast.iter_child_nodes(node):
                if not isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                c_first, c_last = node_char_range(child)
                method_text = content[c_first:c_last]
                method_symbols = extract_symbols(child)
                method_covered.append((c_first, c_last))
                enriched = class_sig + "\n    ...\n" + method_text.strip()

                if len(enriched) <= max_chunk_size:
                    chunks.append(Chunk(
                        text=enriched,
                        file_path=file_path,
                        first_character_index=c_first,
                        last_character_index=c_last,
                        chunk_type="code",
                        symbols=method_symbols,
                    ))
                else:
                    chunks.extend(split_by_size(
                        enriched, file_path, "code",
                        max_chunk_size, c_first, method_symbols,
                    ))

            if not method_covered:
                chunks.extend(split_by_size(
                    node_text, file_path, "code",
                    max_chunk_size, first, symbols,
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
    chunks: list[Chunk] = []
    lines = content.splitlines(keepends=True)

    section_lines: list[str] = []
    section_start_offset = 0
    char_offset = 0

    # Stack index level
    title_stack: list[str] = [""] * 7

    def current_breadcrumb() -> str:
        return " > ".join(t for t in title_stack if t)

    def header_level(line: str) -> int:
        count = 0
        for ch in line:
            if ch == "#":
                count += 1
            else:
                break
        return count if line.startswith("#") else 0

    def is_table_line(line: str) -> bool:
        return line.strip().startswith("|")

    def flush(end_offset: int) -> None:
        if not section_lines:
            return

        breadcrumb = current_breadcrumb()

        blocks: list[tuple[str, bool]] = []
        current_block: list[str] = []
        current_is_table = is_table_line(section_lines[0])

        for line in section_lines:
            line_is_table = is_table_line(line)
            if line_is_table != current_is_table:
                if current_block:
                    blocks.append(("".join(current_block), current_is_table))
                current_block = [line]
                current_is_table = line_is_table
            else:
                current_block.append(line)
        if current_block:
            blocks.append(("".join(current_block), current_is_table))

        block_offset = section_start_offset
        for block_text, is_table in blocks:
            block_stripped = block_text.strip()
            if not block_stripped:
                block_offset += len(block_text)
                continue

            block_end = block_offset + len(block_text)

            if is_table:
                full_text = (
                    breadcrumb + "\n" + block_stripped
                    if breadcrumb else block_stripped
                )
                chunks.append(Chunk(
                    text=full_text,
                    file_path=file_path,
                    first_character_index=block_offset,
                    last_character_index=block_end,
                    chunk_type="doc",
                ))
            else:
                prefixed = (
                    breadcrumb + "\n" + block_stripped
                    if breadcrumb else block_stripped
                )
                if len(prefixed) <= max_chunk_size:
                    chunks.append(Chunk(
                        text=prefixed,
                        file_path=file_path,
                        first_character_index=block_offset,
                        last_character_index=block_end,
                        chunk_type="doc",
                    ))
                else:
                    chunks.extend(split_by_size(
                        prefixed, file_path, "doc",
                        max_chunk_size, block_offset,
                    ))

            block_offset += len(block_text)

    for line in lines:
        level = header_level(line)
        if level > 0 and section_lines:
            flush(char_offset)
            section_lines = []
            section_start_offset = char_offset
            # Actualize the stack
            title_stack[level] = line.strip()
            for i in range(level + 1, 7):
                title_stack[i] = ""
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
    PARENT_CHUNKS_FILE = "parent_chunks.json"

    def __init__(self, repo_root: str,
                 code_chunk_size: int = DEFAULT_CODE_CHUNK_SIZE,
                 doc_chunk_size: int = DEFAULT_DOC_CHUNK_SIZE,
                 code_child_size: int = DEFAULT_CODE_CHILD_SIZE,
                 doc_child_size: int = DEFAULT_DOC_CHILD_SIZE) -> None:
        """Initialise the Ingester.

        Args:
            repo_root: Path to the vLLM repository root directory.
            code_chunk_size: Maximum characters per code chunk.
            doc_chunk_size: Maximum characters per doc chunk.
        """
        self.repo_root = repo_root

        self.code_chunk_size = code_chunk_size
        self.code_child_size = code_child_size

        self.doc_chunk_size = doc_chunk_size
        self.doc_child_size = doc_child_size

        self.chunks: list[Chunk] = []
        self.parent_chunks: list[Chunk] = []

        self.bm25: Optional[bm25s.BM25] = None

        self.embeddings: Optional[np.ndarray] = None

    def build(self, use_embeddings: bool = True) -> None:
        """Ingest the repository and fit the BM25 index.

        Args:
            use_embeddings: When True, also encodes all chunks with
                sentence-transformers for semantic search.
        """

        self.parent_chunks = self.collect_chunks(
            code_size=self.code_chunk_size,
            doc_size=self.doc_chunk_size)
        self.chunks = self.collect_child_chunks()
        self.bm25 = self.build_bm25(self.chunks)

        if use_embeddings:
            self.embeddings = self.build_embeddings(self.parent_chunks)

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

        parent_chunks_path = os.path.join(chunks_dir, self.PARENT_CHUNKS_FILE)
        with open(parent_chunks_path, "w", encoding="utf-8") as fh:
            json.dump(
                [c.to_dict() for c in self.parent_chunks],
                fh, ensure_ascii=False
            )

        doc_count = sum(1 for c in self.chunks if c.chunk_type == "doc")
        code_count = sum(1 for c in self.chunks if c.chunk_type == "code")
        print(
            f"Saved {len(self.chunks)} chunks "
            f"({doc_count} doc, {code_count} code)"
        )
        print(f"Saved {len(self.parent_chunks)} parent chunks")

        if self.embeddings is not None:
            embed_path = os.path.join(output_dir, self.EMBEDDINGS_FILE)
            np.save(embed_path, self.embeddings)
            print(f"Saved embeddings {self.embeddings.shape} -> {embed_path}")

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
        with open(os.path.join(chunks_dir, cls.CHUNKS_FILE),
                  "r", encoding="utf-8") as fh:
            ingester.chunks = [Chunk.from_dict(d) for d in json.load(fh)]

        parent_path = os.path.join(chunks_dir, cls.PARENT_CHUNKS_FILE)
        if os.path.isfile(parent_path):
            with open(parent_path, "r", encoding="utf-8") as fh:
                ingester.parent_chunks = [
                    Chunk.from_dict(d) for d in json.load(fh)
                ]
        else:
            ingester.parent_chunks = []

        index_dir = os.path.join(processed_dir, cls.INDEX_DIR)
        ingester.bm25 = bm25s.BM25.load(index_dir, load_corpus=False)

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

    def collect_chunks(self,
                       code_size: int = DEFAULT_CODE_CHUNK_SIZE,
                       doc_size: int = DEFAULT_DOC_CHUNK_SIZE
                       ) -> list[Chunk]:
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
                file_chunks = chunk_python_file(rel_path, content, code_size)
            elif ext in DOC_EXTENSIONS:
                file_chunks = chunk_doc_file(rel_path, content, doc_size)
            else:
                file_chunks = chunk_text_file(rel_path, content, doc_size)

            all_chunks.extend(file_chunks)

        return all_chunks

    def collect_child_chunks(self) -> list[Chunk]:
        children: list[Chunk] = []

        for parent_idx, parent in enumerate(self.parent_chunks):
            _, ext = os.path.splitext(parent.file_path)
            if ext in CODE_EXTENSIONS:
                child_size = self.code_child_size
            else:
                child_size = self.doc_child_size

            if len(parent.text) <= child_size:
                child = Chunk(
                    text=parent.text,
                    file_path=parent.file_path,
                    first_character_index=parent.first_character_index,
                    last_character_index=parent.last_character_index,
                    chunk_type=parent.chunk_type,
                    symbols=parent.symbols,
                    parent_id=parent_idx,
                )
                children.append(child)
            else:
                # Detectar si el padre es una tabla
                first_line = parent.text.split("\n")[0] if parent.text else ""
                is_table = first_line.strip().startswith("|")
                
                # Extraer header de tabla (primeras 2 líneas: headers + separador)
                table_header = ""
                if is_table:
                    table_lines = parent.text.split("\n")
                    header_lines = [line for line in table_lines[:3] if line.strip().startswith("|")]
                    table_header = "\n".join(header_lines) + "\n" if header_lines else ""

                sub_chunks = split_by_size(
                    parent.text,
                    parent.file_path,
                    parent.chunk_type,
                    child_size,
                    parent.first_character_index,
                    parent.symbols,
                )
                for i, sub in enumerate(sub_chunks):
                    # Añadir header de tabla a los hijos que no son el primero
                    if is_table and i > 0 and table_header:
                        sub.text = table_header + sub.text
                    sub.parent_id = parent_idx
                    children.append(sub)

        return children

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
        corpus = [self.processed(str(Path(c.file_path).with_suffix("")))
                  + " " + c.text for c in chunks]
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

        from sentence_transformers import SentenceTransformer

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

    @staticmethod
    def processed(text: str) -> str:
        """Normalise text for BM25 indexing.

        Lowercases, replaces underscores and hyphens with spaces,
        and collapses whitespace.

        Args:
            text: Raw text to normalise.

        Returns:
            Normalised string.
        """
        text = text.lower()
        text = text.replace("_", " ")
        text = text.replace("-", " ")
        return " ".join(text.split())
