PY        := python3
UV        := uv
USER_NAME := $(shell whoami)
BASE      := /goinfre/$(USER_NAME)/rag_cache
ARGS      ?=
VENV      := .venv
CACHE     := $(BASE)/uv_cache
TMP       := $(BASE)/tmp
HF        := $(BASE)/hf_cache
ENV       := HF_HOME=$(HF) \
			 TRANSFORMERS_CACHE=$(HF) \
			 HF_DATASETS_CACHE=$(HF) \
			 UV_CACHE_DIR=$(CACHE) \
			 TMPDIR=$(TMP) \

install:
	@echo "Generating folder structure..."
	@rm -rf data/
	@mkdir -p $(BASE) $(CACHE) $(TMP) $(HF)
	@mkdir -p data/raw \
			 data/processed \
			 data/output \
			 data/datasets
	@mkdir -p .streamlit

	@echo "Downloading physical dependencies from web..."
	@wget https://cdn.intra.42.fr/document/document/49030/datasets_public.zip
	@wget https://cdn.intra.42.fr/document/document/49032/vllm-0.10.1.zip

	@echo "Placing archieves in their locations..."
	@unzip datasets_public.zip
	@unzip vllm-0.10.1.zip
	@touch .streamlit/config.toml
	@printf '[server]\nfileWatcherType = "none"\n\n[browser]\ngatherUsageStats = false\n' > .streamlit/config.toml

	@mv vllm-0.10.1 ./data/raw
	@mv datasets_public/public/AnsweredQuestions ./data/datasets
	@mv datasets_public/public/UnansweredQuestions ./data/datasets

	@echo "Cleaning downloaded archives..."
	@rm -rf datasets_public
	@rm -rf datasets_public.zip
	@rm -rf vllm-0.10.1.zip

	@echo "Creating Virtual Environment and installing dependencies"
	$(ENV) $(UV) venv $(VENV)
	$(ENV) $(UV) sync

run:
	$(ENV) $(UV) run --project . $(PY) -m src $(ARGS)

web:
	$(ENV) $(UV) run streamlit run src/web.py &
	@echo "Web running at http://localhost:8501"

web-stop:
	@echo "Closing web."
	@pkill -f "streamlit run" || true

index:
	$(ENV) $(UV) run --project . $(PY) -m src index \
		--max_chunk_size 2000 --use_embeddings False

index-embed:
	$(ENV) $(UV) run --project . $(PY) -m src index \
		--max_chunk_size 2000 --use_embeddings True

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
	$(ENV) $(UV) run --project . flake8 src
	$(ENV) $(UV) run --project . mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

clean:
	@echo "Cleaning venv and donwloaded data."
	@rm -rf $(BASE)
	@rm -rf $(VENV)
	@rm -rf .mypy_cache
	@rm -rf data
	@rm -rf .streamlit
	@find . -type d -name "__pycache__" -exec rm -rf {} +

meme:
	@echo "Killing in the name of... bad retrieval scores"
	@xdg-open "https://www.youtube.com/watch?v=bWXazVhlyxQ" 2>/dev/null & true
	@echo "https://www.youtube.com/watch?v=bWXazVhlyxQ"
	$(ENV) $(UV) run streamlit run src/web.py &
	@echo "Web running at http://localhost:8501"

.PHONY: install run web web-stop index index-fast search-docs search-code \
        eval-docs eval-code lint clean meme