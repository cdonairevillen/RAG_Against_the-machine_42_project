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
# Fast index without embeddings (~15s, used for evaluation)
make index

# Full index with semantic embeddings (~10 min on CPU, optional bonus)
make index-embed
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
 [Ingester]  — chunks files, builds BM25 + subindices + optional embeddings
      │
      ▼
 [Retriever] — searches indices, applies query expansion, reranking
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
strategies, and builds three BM25 indices (general, docs-only, code-only)
plus an optional semantic embedding index.

**Retriever** exposes two search modes: a fast BM25 path for dataset
evaluation (`search()`), and a precise path with cross-encoder reranking
for LLM generation (`search_for_generation()`). Also implements
`search_with_type_fallback()` and `search_triple_rrf()` as experimental
alternatives (see Design Decisions).

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

Two distinct chunking strategies are applied depending on file type,
with 20% overlap between consecutive chunks to avoid losing context
at boundaries.

### Python files (`.py`, `.pyi`) — AST-based chunking

The file is parsed with Python's `ast` module. Top-level functions and
classes are extracted as individual chunks, preserving complete syntactic
units. This guarantees the LLM always receives valid, meaningful code.

If a unit exceeds `code_chunk_size` (default: 1200 characters), it is
further split with overlap. Module-level code (imports, constants) is
collected as a preamble chunk.

AST symbol names (class names, method names, attribute names) are stored
in a separate `Chunk.symbols` field and used for query expansion at
search time. They are **not** included in the BM25 corpus to avoid
inflating chunk length and penalising BM25 length normalisation.

### Markdown and RST files (`.md`, `.rst`, `.txt`) — Header-based chunking

Files are split on header lines (`#`). Each section from one header to
the next becomes one chunk. Sections exceeding `doc_chunk_size`
(default: 2000 characters) are further split with overlap.

### Fallback — Size-based chunking

Any readable file not matching the above extensions is split into
overlapping chunks of `doc_chunk_size` characters.

**Chunk size configuration:**

```bash
uv run python -m src index --code_chunk_size 1200 --doc_chunk_size 2000
```

---

## Retrieval Method

### BM25 (primary)

The system uses [bm25s](https://github.com/xhluca/bm25s) for all primary
retrieval. BM25 improves on TF-IDF with two corrections:

1. **TF saturation** — repeated terms have diminishing returns, preventing
   documents that simply repeat a term from dominating the ranking.
2. **Length normalisation** — longer documents are penalised so they do
   not win purely by volume.

Three separate BM25 indices are maintained:
- **General index** — all chunks, used for fast evaluation.
- **Docs index** — documentation chunks only.
- **Code index** — code chunks only.

### Query Expansion

Before every BM25 search, the query is expanded with identifier variants.
CamelCase names are decomposed (`PagedAttention` → `Paged Attention
paged_attention`) and snake_case identifiers are split into parts. Symbol
names from matching chunks are also appended, allowing BM25 to match
queries that reference specific class or method names.

### Triple-index search (experimental)

`search_with_type_fallback` detects query type from the top-3 general
results and routes to the matching subindex. `search_triple_rrf` fuses
all three indices with weighted RRF. Both are disabled during evaluation
due to the corpus imbalance (93% code) — without a reranker to correct
misclassifications, routing hurts docs recall. Available for interactive
use where the reranker corrects bias.

### Semantic embeddings (optional bonus)

When enabled with `--use_embeddings True`, the system encodes all chunks
with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, L2-normalised)
and fuses BM25 and cosine similarity rankings with Reciprocal Rank Fusion
(RRF, k=60). Embeddings improve code recall (+7%) but add ~10 minutes to
indexing time, so they are disabled by default.

### Cross-encoder reranking (for generation only)

When answering questions, the system retrieves k×5 candidates with BM25,
then reranks them with `cross-encoder/ms-marco-MiniLM-L-6-v2`. The
cross-encoder scores each (query, chunk) pair jointly, producing a
relevance score more accurate than BM25 alone. Batch reranking is used
for efficiency — all pairs are scored in a single forward pass. This
step is skipped during dataset evaluation to meet the 90-second
throughput requirement.

### Relevance filtering

A minimum BM25 score threshold (4.5) is applied before LLM generation.
Queries with no relevant content in the corpus (score < 4.5) return
an empty result without invoking the LLM.

---

## Performance Analysis

Results on the private evaluation datasets (Recall@5):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---------|----------|----------|----------|-----------|
| Docs    | 0.530    | 0.730    | 0.800    | 0.860     |
| Code    | 0.250    | 0.480    | 0.530    | 0.590     |

Both datasets exceed the mandatory thresholds (≥0.80 for docs, ≥0.50
for code).

**Performance constraints met:**
- Indexing time: 13s (BM25 only, limit 300s)
- Warm retrieval throughput: 200 questions in 23s (limit 90s)

---

## Design Decisions

**Single unified BM25 index for evaluation.** Three BM25 indices are
built (general, docs, code) but evaluation uses only the general index.
Experiments with type-based routing (majority vote on top-3 BM25 results)
showed that the corpus imbalance (1,701 doc vs 25,085 code chunks, i.e.
93% code) causes almost all queries to be routed to the code subindex,
collapsing docs recall from 0.800 to 0.700. The subindices are used in
`search_with_type_fallback` and `search_triple_rrf` for interactive use,
where the cross-encoder reranker corrects misclassifications.

**AST chunking over line-based splitting.** Cutting a function in half
destroys its meaning. AST extraction guarantees the LLM always receives
complete, syntactically valid units. The performance cost is negligible
compared to the quality gain.

**Symbols stored separately from corpus text.** Adding AST symbol names
directly to chunk text inflated chunk length and penalised BM25 length
normalisation, reducing recall. Storing them in a separate field and
using them only for query expansion avoids this penalty.

**Reranker only for generation, not evaluation.** The cross-encoder
reranker improves retrieval quality (Docs @5: 0.800 → 0.820) but
processing 50 candidates per query takes ~2 seconds on CPU. For the
200-question throughput test this would exceed the 90-second limit.
The reranker is applied only when generating answers, using batch
prediction to score all pairs in a single forward pass for efficiency.

**Native `model.generate()` instead of manual greedy decoding.** The
SDK's `get_logits_from_input_ids()` recomputes all previous tokens on
every step (O(n²) complexity, no KV-cache). Accessing the underlying
HuggingFace model's `generate()` method directly reduces generation
time from minutes to seconds per answer.

**Qwen3 chat format with thinking disabled.** Qwen3-0.6B has a built-in
reasoning mode that emits `<think>` blocks before answering. Using the
correct chat template with `/no_think` in the user message and stripping
residual `<think>` blocks with regex prevents the model from outputting
reasoning traces in the final answer.

---

## Challenges Faced

**Windows vs Linux path separators.** During development on Windows,
`os.path.relpath()` returns backslash-separated paths. The ground-truth
dataset uses forward slashes. This caused all evaluations to show 0.000
recall until a `.replace("\\", "/")` normalisation was added to all
path assignments.

**Double nesting in llm_sdk.** The SDK package structure required
`from llm_sdk import Small_LLM_Model` rather than the initially assumed
`from llm_sdk.llm_sdk import Small_LLM_Model`. Misidentifying the import
path caused import errors that were not immediately obvious.

**Qwen3 thinking mode.** Qwen3-0.6B defaults to emitting a reasoning
chain before answering, producing responses like `<think>... </think>
answer`. This was resolved by adding `/no_think` to the prompt and
post-processing output with a regex strip.

**Cache space on 42 clusters.** The home directory on 42 machines has
very limited space (~28 MB free). All HuggingFace model weights, uv
package cache, and temporary files must be redirected to `/goinfre` via
environment variables. The Makefile handles this automatically.

**BM25 score calibration for relevance filtering.** The initial threshold
of 1.0 was too low — common English words like "you", "are", "stupid"
scored 2.7-3.3 in the vLLM corpus. Empirical testing across valid and
invalid queries established 4.5 as a reliable threshold that separates
relevant queries (score ≥ 5.3) from irrelevant ones (score ≤ 3.3).

**Embeddings degrading docs recall.** The generic `all-MiniLM-L6-v2`
model improves code recall (+7%) but slightly reduces docs recall from
0.800 to 0.820 — a net tradeoff. More critically, encoding 26,786 chunks
takes ~10 minutes, exceeding the 300-second indexing limit. Embeddings
are therefore optional and disabled by default.

**Triple indexation with imbalanced corpus.** Routing queries to doc or
code subindices based on BM25 top-k majority vote failed because 93% of
chunks are code — almost all queries were routed to the code subindex
regardless of type, collapsing docs recall from 0.800 to 0.700.
Variants tested: top-1, top-3, top-7, `doc_count >= 1`, RRF with dynamic
weights. All variants either hurt docs or hurt code. The approach works
when the cross-encoder reranker corrects misclassifications but violates
the throughput constraint during evaluation.

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
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — embedding model

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