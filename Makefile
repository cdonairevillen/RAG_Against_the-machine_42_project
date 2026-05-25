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
	$(ENV) $(UV) run --project . $(PY) -m src $(ARGS)

debug:
	$(ENV) $(UV) run --project . $(PY) -m pdb -m src $(ARGS)

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

# ______________________________Pipeline___________________________________

index:
	$(ENV) $(UV) run --project . $(PY) -m src index --max_chunk_size 2000

index-fast:
	$(ENV) $(UV) run --project . $(PY) -m src index --max_chunk_size 2000 --use_embeddings False

search-docs:
	$(ENV) $(UV) run --project . $(PY) -m src answer_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--skip_generation True

search-code:
	$(ENV) $(UV) run --project . $(PY) -m src answer_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
		--skip_generation True

eval-docs:
	$(ENV) $(UV) run --project . $(PY) -m src evaluate \
		--student_answer_path data/output/answers/dataset_docs_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
		--k 10

eval-code:
	$(ENV) $(UV) run --project . $(PY) -m src evaluate \
		--student_answer_path data/output/answers/dataset_code_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
		--k 10

.PHONY: install update run debug lint lint-strict clean re index \
        search-docs search-code eval-docs eval-code