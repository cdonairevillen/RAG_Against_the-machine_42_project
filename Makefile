PY        := python3
UV        := uv
USER_NAME := $(shell whoami)
BASE      := /goinfre/$(USER_NAME)/rag_cache

ARGS ?=

VENV  := .venv
CACHE := $(BASE)/uv_cache
TMP   := $(BASE)/tmp
HF    := $(BASE)/hf_cache

ENV := \
    HF_HOME=$(HF) \
    TRANSFORMERS_CACHE=$(HF) \
    HF_DATASETS_CACHE=$(HF) \
    UV_CACHE_DIR=$(CACHE) \
    TMPDIR=$(TMP)


install:
    mkdir -p $(BASE) $(CACHE) $(TMP) $(HF)
    $(ENV) $(UV) venv $(VENV)
    $(ENV) $(UV) sync

update:
    $(ENV) $(UV) sync --reinstall

run:
    $(ENV) $(UV) run --project . $(PY) -m student $(ARGS)

# Ejemplos de uso:
#   make run ARGS="index --max_chunk_size 2000"
#   make run ARGS="search 'How to configure OpenAI server?' --k 10"
#   make run ARGS="answer 'How to configure OpenAI server?' --k 10"
#   make run ARGS="search_dataset --dataset_path data/datasets/..."
#   make run ARGS="answer_dataset --student_search_results_path ..."
#   make run ARGS="evaluate --student_answer_path ... --dataset_path ..."

debug:
    $(ENV) $(UV) run --project . $(PY) -m pdb -m student $(ARGS)

lint:
    $(ENV) $(UV) run --project . flake8 src
    $(ENV) $(UV) run --project . mypy src \
        --warn-return-any \
        --warn-unused-ignores \
        --ignore-missing-imports \
        --disallow-untyped-defs \
        --check-untyped-defs

lint-strict:
    $(ENV) $(UV) run --project . flake8 src
    $(ENV) $(UV) run --project . mypy src --strict


clean:
    rm -rf $(BASE)
    rm -rf $(VENV)
    rm -rf .mypy_cache
    find . -type d -name "__pycache__" -exec rm -rf {} +

re: clean install

index:
    $(ENV) $(UV) run --project . $(PY) -m student index --max_chunk_size 2000

eval-docs:
    $(ENV) $(UV) run --project . $(PY) -m student evaluate \
        --student_answer_path data/output/search_results/dataset_docs_public.json \
        --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
        --k 10

eval-code:
    $(ENV) $(UV) run --project . $(PY) -m student evaluate \
        --student_answer_path data/output/search_results/dataset_code_public.json \
        --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
        --k 10

.PHONY: install update run debug lint lint-strict clean re index eval-docs eval-code
Copy failed — try from claude.ai in browser