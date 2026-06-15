uv run python -m src index --code_chunk_size 1200 --doc_chunk_size 2000 --use_embeddings False    

uv run python -m src answer_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --skip_generation True --save_directory data/output/search_results

uv run python -m src evaluate --student_answer_path data/output/search_results/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10

uv run python -m src answer_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --skip_generation True --save_directory data/output/search_results

uv run python -m src evaluate --student_answer_path data/output/search_results/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --k 10