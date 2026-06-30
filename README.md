*This project has been created as part of the 42 curriculum by cdonaire*

# RAG against the Machine

---

## Description

**RAG against the Machine** is a Retrieval-Augmented Generation system built
to answer questions about the [vLLM](https://github.com/vllm-project/vllm)
codebase. Given a natural-language question, the system retrieves the most
relevant code snippets and documentation from the vLLM repository and uses
a local language model (Qwen/Qwen3-0.6B) to generate a grounded, faithful
answer.

The system is evaluated on its ability to correctly retrieve source locations
when asked questions about vLLM, and to generate accurate answers based
exclusively on retrieved context — without hallucinating information not
present in the sources.

---

## Instructions

### Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) as package and project manager
- The vLLM 0.10.1 repository (provided as attachment by the subject)

### Installation

```bash
# Clone the repository and enter the project directory
git clone <repo-url>
cd rag_against_the_machine

# Install all dependencies (redirects cache to goinfre on 42 clusters)
make install
```

### Indexing the knowledge base

Before running any queries, the vLLM repository must be indexed.
Place the unzipped repository at `data/raw/vllm-0.10.1/` and run:

```bash
make index
```

### Running the system

```bash
# Answer a single question
make run ARGS="answer 'How does PagedAttention work?' --k 5"

# Search without generating an answer
make run ARGS="search 'How does PagedAttention work?' --k 10"

# Run retrieval over a full dataset
make search-docs
make search-code

# Evaluate recall@k against ground truth
make eval-docs
make eval-code

# Launch the web interface
make web
```

### Running via CLI directly

```bash
uv run python -m src index
uv run python -m src search "How to configure OpenAI server?" --k 10
uv run python -m src answer "How to configure OpenAI server?" --k 10
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
uv run python -m src answer_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
uv run python -m src evaluate \
    --student_answer_path data/output/search_results/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

---

## System Architecture

The pipeline is composed of five main components:

```
vLLM Repository
      │
      ▼
 [Ingester]  — chunks files, builds three BM25 indices
      │
      ▼
 [Retriever] — routes queries to the right index, reranks when needed
      │
      ▼
 [Generator] — builds prompt, runs Qwen3-0.6B, returns answer
      │
      ▼
 [Evaluator] — computes Recall@k with 5% character overlap threshold
      │
      ▼
   [CLI]     — orchestrates all components via Python Fire
```

**Ingester** reads the vLLM repository, applies type-specific chunking
strategies, and builds three BM25 indices: a unified index over all
chunks, a docs-only index, and a code-only index.

**Retriever** exposes a classifier-guided search mode
(`search_smart()`) that routes each query to whichever per-type index
it best matches, plus a precise path with cross-encoder reranking for
LLM generation (`search_for_generation()`). A fast BM25-only path
(`search()`) is also available for simple lookups.

**Generator** uses Qwen/Qwen3-0.6B via HuggingFace Transformers with
a strict RAG prompt that prevents hallucination. Generation uses greedy
decoding via the native `model.generate()` for KV-cache efficiency.

**Evaluator** computes Recall@k metrics where a retrieved source counts
as a hit when it overlaps at least 5% with any ground-truth source,
measured in characters.

**CLI** is a single `CLI` class that lazily loads the Retriever
and Generator on first use and reuses them across all commands.

---

## Chunking Strategy

Chunks are built in two layers — **parents** and **children** — and
two distinct chunking strategies are applied at the parent level
depending on file type. Both strategies use 20% overlap when a unit
must be split further, to avoid losing context at boundaries.

### Python files (`.py`, `.pyi`) — AST-based parent chunking

The file is parsed with Python's `ast` module. Top-level functions and
classes are extracted as individual parent chunks, preserving complete
syntactic units. This guarantees the retriever always works with valid,
meaningful code rather than an arbitrary line slice.

If a unit exceeds `code_chunk_size` (default: 2000 characters), it is
further split with overlap. For large classes, each method becomes its
own chunk prefixed with the class signature, so methods stay
identifiable even out of context. Module-level code (imports,
constants) is collected as a preamble chunk.

AST symbol names (class names, method names, attribute names) are
stored in a separate `Chunk.symbols` field and used for query
expansion at search time. They are **not** included in the BM25 corpus
to avoid inflating chunk length and penalising BM25 length
normalisation.

### Markdown and RST files (`.md`, `.rst`, `.txt`) — header-based parent chunking

Files are split on header lines (`#`). Sections smaller than 200
characters are accumulated together with the next section before being
flushed as a chunk — a lone `# TPU` header with nothing else under it
would otherwise become a near-empty parent that downstream ranking
cannot work with. Tables are kept as indivisible units and prefixed
with their section title so a table fragment retains its context.
Sections exceeding `doc_chunk_size` (default: 2000 characters) are
split with overlap.

### Fallback — size-based chunking

Any readable file not matching the above extensions is split into
overlapping chunks of `doc_chunk_size` characters.

### Children

Each parent is further split into children (max 600 characters) used
for BM25 indexing. Children give BM25 precise, low-noise matches;
once a child matches, its parent — which carries the full surrounding
context — is what gets returned and passed to the reranker or the LLM.
Table continuation children get the table header re-attached so a
mid-table child chunk is still interpretable on its own.

**Chunk size configuration:**

```bash
uv run python -m src index --code_chunk_size 2000 --doc_chunk_size 2000
```

---

## Retrieval Method

### BM25 (primary)

The system uses [bm25s](https://github.com/xhluca/bm25s) for all
retrieval. BM25 improves on TF-IDF with two corrections:

1. **TF saturation** — repeated terms have diminishing returns,
   preventing documents that simply repeat a term from dominating the
   ranking.
2. **Length normalisation** — longer documents are penalised so they
   do not win purely by volume.

Three separate BM25 indices are maintained over child chunks:
- **General index** — all chunks, used for the unfiltered fast path
  and as the candidate pool for code-mode queries.
- **Docs index** — documentation chunks only.
- **Code index** — code chunks only.

### Query Expansion

Before every BM25 search, the query is expanded with identifier
variants. CamelCase names are decomposed (`PagedAttention` → `Paged
Attention paged_attention`) and snake_case identifiers are split into
parts. Symbol names from matching chunks are also appended, allowing
BM25 to match queries that reference specific class or method names.

### Classifier-guided routing — `search_smart()`

This is the retrieval mode used for both evaluation and the web
interface. Each query is scored against the doc index and the code
index independently (mean BM25 score of the top 3 hits in each),
combined with lexical signals — identifier-like tokens (snake_case,
CamelCase, code keywords) push the query toward code; natural-language
markers (`how`, `configure`, `hardware`, `cli`...) push it toward docs.

- **Doc-leaning queries** are answered with plain BM25 over the doc
  index. No reranker. The cross-encoder consistently ranked code
  chunks above the correct doc chunk on this corpus, so skipping it
  here was a measured improvement, not a shortcut.
- **Code-leaning queries** are answered from the general index
  filtered down to code chunks, then reranked with the cross-encoder
  — unless the BM25 signal is already unambiguous (score above 9.0),
  in which case the reranker is skipped to save time. A handful of
  doc chunks are always mixed into the candidate pool as a safety net
  for queries the classifier got slightly wrong.

### Cross-encoder reranking — `cross-encoder/ms-marco-MiniLM-L-6-v2`

Used in two places: inside `search_smart()` for ambiguous code
queries, and in `search_for_generation()`, which reranks the combined
doc+code candidate pool and guarantees at least two doc results
survive into the final context whenever docs were present in the
pool. The cross-encoder scores each (query, chunk) pair jointly,
producing a relevance score more accurate than BM25 alone.

### Relevance filtering

A minimum BM25 score threshold (5.5) is applied before LLM generation
in `search_for_generation()`. Queries with no relevant content in the
corpus return an empty result without invoking the LLM, so the system
answers "Not found in the provided sources" instead of guessing.

---

## Performance Analysis

Results on the private evaluation datasets:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---------|----------|----------|----------|-----------|
| Docs    | 0.700    | 0.850    | 0.880    | 0.890     |
| Code    | 0.440    | 0.630    | 0.670    | 0.750     |

Both datasets clear the mandatory thresholds.

**Performance constraints met:**
- Indexing time: well under the limit (BM25-only, no embeddings).
- Code dataset retrieval: 100 questions in ~45s, with the
  cross-encoder loaded — comfortably inside the throughput window.

---

## Design Decisions

**Per-type BM25 indices instead of one unified index.** Building
separate doc and code indices, rather than relying on the general
index alone, was the single biggest improvement to docs recall. The
vLLM corpus is heavily code-dominated, so any unified ranking
naturally favours code chunks even for clearly documentation-style
questions; isolating the doc index removes that competition entirely.

**Lexical + BM25 classifier over a pure score comparison.** Comparing
raw BM25 scores between the doc and code index alone misclassified
queries that read as natural language but target a specific function
or parameter (`"What is the default value of the eps parameter in
RMSNorm?"`). Adding a count of identifier-like tokens (snake_case,
CamelCase, code keywords) versus natural-language markers
substantially reduced these misclassifications without needing a
trained model.

**Reranker only where it helps, not everywhere.** Initial testing
applied the cross-encoder uniformly across both datasets. It improved
code recall but consistently *hurt* docs recall — short, natural
doc questions were outranked by denser code chunks. The fix was not
to drop the reranker but to scope it: skip it entirely for doc-mode
queries, and skip it even for code-mode queries once BM25 is already
confident (score > 9.0), since reranking a list that BM25 has already
sorted correctly only adds latency.

**AST chunking over line-based splitting.** Cutting a function in half
destroys its meaning. AST extraction guarantees retrieval always works
with complete, syntactically valid units. The performance cost is
negligible compared to the quality gain.

**Symbols stored separately from corpus text.** Adding AST symbol
names directly to chunk text inflated chunk length and penalised BM25
length normalisation, reducing recall. Storing them in a separate
field and using them only for query expansion avoids this penalty.

**Accumulating small doc sections before flushing a chunk.** Early
header-based chunking produced parent chunks as small as 5 characters
for files like `tpu.md`, where a header is followed immediately by
another header. These near-empty chunks gave the reranker nothing to
work with and were effectively unretrievable. Accumulating sections
under 200 characters into the next section before flushing fixed this
without changing the semantic chunk boundaries that matter.

**Native `model.generate()` instead of manual greedy decoding.** The
SDK's `get_logits_from_input_ids()` recomputes all previous tokens on
every step (O(n²) complexity, no KV-cache). Accessing the underlying
HuggingFace model's `generate()` method directly reduces generation
time from minutes to seconds per answer.

**Qwen3 chat format with thinking disabled.** Qwen3-0.6B has a
built-in reasoning mode that emits `<think>` blocks before answering.
Using the correct chat template with `/no_think` in the user message
and stripping residual `<think>` blocks with regex prevents the model
from outputting reasoning traces in the final answer.

**Embeddings explored and dropped.** Semantic embeddings
(`all-MiniLM-L6-v2`, fused with BM25 via Reciprocal Rank Fusion) were
tested both over the full corpus and restricted to doc chunks only.
In every configuration tested, they reduced docs recall rather than
improving it, on top of adding meaningful indexing time. They were
removed from the pipeline entirely rather than kept as an underused
option.

---

## Challenges Faced

**Windows vs Linux path separators.** During development on Windows,
`os.path.relpath()` returns backslash-separated paths. The
ground-truth dataset uses forward slashes. This caused evaluations to
show near-zero recall until a `.replace("\\", "/")` normalisation was
added to all path assignments.

**Qwen3 thinking mode.** Qwen3-0.6B defaults to emitting a reasoning
chain before answering, producing responses like `<think>... </think>
answer`. This was resolved by adding `/no_think` to the prompt and
post-processing output with a regex strip.

**Cache space on 42 clusters.** The home directory on 42 machines has
very limited space. All HuggingFace model weights, uv package cache,
and temporary files must be redirected to `/goinfre` via environment
variables. The Makefile handles this automatically.

**BM25 score calibration for relevance filtering.** The initial
threshold was too low — common English phrasing alone could score
above it in the vLLM corpus. Empirical testing across valid and
invalid queries established 5.5 as a reliable threshold that filters
out-of-domain questions without rejecting legitimate ones.

**The reranker hurting recall on docs.** The intuitive assumption —
"a precision reranker can only help" — turned out to be wrong for
this corpus. Diagnosing this required comparing recall with and
without the reranker on both datasets separately rather than trusting
the combined number, which made the effect on docs invisible.

**Near-empty parent chunks from header-only sections.** Files where a
top-level header is immediately followed by another header (no body
text in between) produced parent chunks of a handful of characters.
These chunks technically existed and could be retrieved by BM25, but
carried no usable context for the reranker or the LLM. Diagnosing this
required tracing individual ground-truth misses back to their parent
chunk text, not just their BM25 rank.

---

```bash
# Build the index
make index

# Answer a question about vLLM internals
make run ARGS="answer 'How does PagedAttention manage KV cache blocks?' --k 5"

# Evaluate retrieval quality on the docs dataset
make search-docs
make eval-docs

# Launch the web interface
make web
# → opens http://localhost:8501
```

**Example answer output:**

```
Prompt: "How does PagedAttention work?"

=== Answer ===
PagedAttention works by dividing the input into blocks, where each block
contains a portion of the sequence and attention weights. The implementation
uses either V1 or V2 kernels depending on context — V1 for smaller sequences
and larger numbers of heads, V2 when context length exceeds 8192 tokens.

=== Sources ===
[1] filepath: data/raw/vllm-0.10.1/vllm/attention/ops/paged_attn.py
    chunk:    0
    range:    4053:5244
    text:     # PagedAttention V1 or V2...
```

---

## Resources

### RAG and Information Retrieval

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — original RAG paper (Lewis et al., 2020)
- [BM25: The Definitive Guide](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) — BM25 algorithm explained
- [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) — RRF paper (Cormack et al., 2009)
- [bm25s documentation](https://github.com/xhluca/bm25s)
- [sentence-transformers documentation](https://www.sbert.net/)
- [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)

### Models

- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — generation model

### Tools

- [vLLM repository](https://github.com/vllm-project/vllm) — the indexed knowledge base
- [Python ast module](https://docs.python.org/3/library/ast.html) — used for AST-based chunking
- [uv package manager](https://github.com/astral-sh/uv)
- [Python Fire](https://github.com/google/python-fire) — CLI framework
- [Streamlit](https://streamlit.io/) — web interface

### AI Usage

AI assistance (Claude, Anthropic) was used throughout this project for:

- **Architecture design** — planning the RAG pipeline structure, class
  responsibilities, and data flow between components.
- **Implementation guidance** — debugging import errors, calibrating BM25
  score thresholds, fixing the Qwen3 chat format and thinking mode issues.
- **Code review** — identifying the Windows/Linux path separator bug,
  the BM25 length normalisation issue caused by embedding symbols in chunk
  text, and the KV-cache performance problem with manual greedy decoding.
- **Documentation** — structuring this README and the inline docstrings
  throughout the codebase.

All AI-generated code was reviewed, understood, tested, and modified
before being included in the final submission. The student is responsible
for all design decisions and can explain every part of the implementation.