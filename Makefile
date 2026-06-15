PY        := python3
UV        := uv
USER_NAME := $(shell whoami)
BASE      := /goinfre/$(USER_NAME)/rag_cache
ARGS      ?=
VENV      := .venv
CACHE     := $(BASE)/uv_cache
TMP       := $(BASE)/tmp
HF        := $(BASE)/hf_cache
ENV       := \
	HF_HOME=$(HF) \
	TRANSFORMERS_CACHE=$(HF) \
	HF_DATASETS_CACHE=$(HF) \
	UV_CACHE_DIR=$(CACHE) \
	TMPDIR=$(TMP)

install:
	mkdir -p $(BASE) $(CACHE) $(TMP) $(HF)
	mkdir -p data/raw data/processed data/output data/datasets
	$(ENV) $(UV) venv $(VENV)

run:
	$(ENV) $(UV) run --project . $(PY) -m src $(ARGS)

web:
	$(ENV) $(UV) run streamlit run src/web.py

index:
	$(ENV) $(UV) run --project . $(PY) -m src index \
		--code_chunk_size 1200 --doc_chunk_size 2000 --use_embeddings True

index-fast:
	$(ENV) $(UV) run --project . $(PY) -m src index \
		--code_chunk_size 1200 --doc_chunk_size 2000 --use_embeddings False

search-docs:
	$(ENV) $(UV) run --project . $(PY) -m src answer_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--skip_generation True \
		--save_directory data/output/search_results

search-code:
	$(ENV) $(UV) run --project . $(PY) -m src answer_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
		--skip_generation True \
		--save_directory data/output/search_results

eval-docs:
	$(ENV) $(UV) run --project . $(PY) -m src evaluate \
		--student_answer_path data/output/search_results/dataset_docs_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
		--k 10

eval-code:
	$(ENV) $(UV) run --project . $(PY) -m src evaluate \
		--student_answer_path data/output/search_results/dataset_code_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
		--k 10

lint:
	$(ENV) $(UV) run --project . flake8 src --max-line-length 84
	$(ENV) $(UV) run --project . mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

clean:
	rm -rf $(BASE)
	rm -rf $(VENV)
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

meme:index-fast web
	@echo "Killing in the name of... bad retrieval scores"
	@xdg-open "https://www.youtube.com/watch?v=bWXazVhlyxQ" 2>/dev/null || \
		open "https://www.youtube.com/watch?v=bWXazVhlyxQ" 2>/dev/null || \
		echo "https://www.youtube.com/watch?v=bWXazVhlyxQ"

.PHONY: install run index-fast search-docs search-code \
        eval-docs eval-code lint clean meme