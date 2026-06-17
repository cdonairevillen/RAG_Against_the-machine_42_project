import os
import re
import torch
from llm_sdk.__init__ import Small_LLM_Model
from src.models import MinimalSource


def build_prompt(question: str, context_blocks: list[str]) -> str:
    """Assemble prompt using Qwen3 chat format with thinking disabled."""
    context_section = "\n\n".join(context_blocks)
    system = (
        "You are a read-only technical documentation assistant for the "
        "vLLM codebase. Your only job is to answer questions using the "
        "SOURCE CODE AND DOCUMENTATION provided below.\n\n"
        "STRICT RULES — follow all of them:\n"
        "1. Use ONLY information explicitly stated in the context. "
        "Never infer, guess, or complete missing information.\n"
        "2. If the context contains a function definition or template "
        "string, do NOT simulate calling it or filling its placeholders. "
        "Code is not data.\n"
        "3. If the context contains example or test code with hardcoded "
        "values, do NOT treat those values as real facts.\n"
        "4. If the answer is not explicitly present in the context, "
        "respond ONLY with exactly this and nothing else: "
        "'Not found in the provided sources.'\n"
        "5. Never use your training knowledge. If you know the answer "
        "but it is not in the context, still respond ONLY with: "
        "'Not found in the provided sources.'\n"
        "6. Keep your answer concise. If you found the answer, always "
        "cite the source file at the end.\n\n"
        "FORMAT when answer IS found: direct answer + 'Source: <filenames>' "
        "<filenames> will be filled with all the files where your answer comes"
        " from\n"
        "FORMAT when answer IS NOT found: 'Not found in the provided sources.'"
        "BEFORE ANSWERING, verify: does the context explicitly contain "
        "the answer to the question? If NO, respond ONLY with: "
        "'Not found in the provided sources.' — do not use any other "
        "knowledge under any circumstances.\n"
    )
    user_content = (
        f"CONTEXT:\n{context_section}\n\nQUESTION: {question} /no_think"
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class Generator:
    """Loads Qwen/Qwen3-0.6B and generates grounded answers via greedy
    decoding.

    Greedy decoding picks the highest-logit token at each step, making
    generation deterministic and faithful to the retrieved context.

    The model is loaded once at construction and reused for all calls.
    Instantiate once and share via the CLI lazy loader.

    Usage:
        generator = Generator()
        answer = generator.answer(
            "How does PagedAttention work?", sources
        )

    Attributes:
        model: Loaded Small_LLM_Model instance.
        max_new_tokens: Maximum tokens to generate per answer.
        repo_root: Used to read chunk text from disk.
    """

    MODEL_NAME = "Qwen/Qwen3-0.6B"

    def __init__(self, max_new_tokens: int = 150,
                 repo_root: str = "data/raw/vllm-0.10.1") -> None:
        """Load the LLM. First run downloads weights (~600 MB).

        Args:
            max_new_tokens: Cap on generated tokens per answer.
                Keep low (100-200) — each token costs a full forward pass.
            repo_root: Root of the vLLM repo for reading chunk text.
        """
        print(f"Loading {self.MODEL_NAME}...")
        self.model = Small_LLM_Model(model_name=self.MODEL_NAME)
        self.max_new_tokens = max_new_tokens
        self.repo_root = repo_root
        print("Model loaded.")

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        """Generate a grounded answer for a question given retrieved sources.

        Args:
            question: The natural-language question.
            sources: Retrieved MinimalSource objects from the Retriever.

        Returns:
            A concise, source-grounded answer string.
        """
        if not question or not question.strip():
            return "No question provided."
        if not sources:
            return "No relevant sources found."

        context_blocks = self.build_context_blocks(sources)
        if not context_blocks:
            return "Could not read source content from disk."

        prompt = build_prompt(question, context_blocks)
        prompt = self.truncate_prompt(prompt)
        return self.greedy_decode(prompt)

    def answer_batch(self, questions: list[str],
                     sources_list: list[list[MinimalSource]]) -> list[str]:
        """Generate answers for multiple questions.

        Args:
            questions: List of question strings.
            sources_list: Parallel list of retrieved sources per question.

        Returns:
            List of answer strings in the same order as input.
        """
        answers: list[str] = []
        for question, sources in zip(questions, sources_list):
            try:
                answers.append(self.answer(question, sources))
            except Exception:
                answers.append("Error generating answer.")
        return answers

    def build_context_blocks(self, sources: list[MinimalSource]) -> list[str]:
        """Read the actual text for each source from disk and format it.

        Args:
            sources: Retrieved source objects with file_path and char indices.

        Returns:
            List of formatted strings: '--- file: path ---\\nchunk text'.
        """
        blocks: list[str] = []
        for source in sources:
            text = self.read_chunk_text(source)
            if text:
                blocks.append(f"--- file: {source.file_path} ---\n{text}")
        return blocks

    def read_chunk_text(self, source: MinimalSource) -> str:
        """Extract the exact text slice from the source file on disk.

        Args:
            source: MinimalSource with file_path and character indices.

        Returns:
            The text slice, or empty string if the file cannot be read.
        """
        abs_path = os.path.join(self.repo_root, source.file_path)
        if not os.path.isfile(abs_path):
            abs_path = source.file_path
        try:
            with open(
                abs_path, "r", encoding="utf-8", errors="ignore"
            ) as fh:
                content = fh.read()
            return content[
                source.first_character_index:source.last_character_index
            ]
        except (OSError, IndexError):
            return ""

    def truncate_prompt(self, prompt: str, max_chars: int = 3000) -> str:
        """Truncate the prompt to stay within a reasonable token budget.

        Keeps the system prompt and question intact; truncates only the
        middle context section so the model always sees the full question.

        Args:
            prompt: Full assembled prompt.
            max_chars: Approximate character budget (1 token ≈ 4 chars).

        Returns:
            Prompt truncated to max_chars if necessary.
        """
        if len(prompt) <= max_chars:
            return prompt
        tail_marker = "QUESTION:"
        tail_idx = prompt.rfind(tail_marker)
        if tail_idx == -1:
            return prompt[:max_chars]
        tail = prompt[tail_idx:]
        budget = max_chars - len(tail)
        return prompt[:budget] + "\n[context truncated]\n" + tail

    def greedy_decode(self, prompt: str) -> str:
        """Generate text using HuggingFace's native generate() with KV-cache.

        Args:
            prompt: The full prompt string to continue from.

        Returns:
            Generated text (new tokens only, prompt excluded).
        """

        inputs = self.model._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model._device)

        eos_token = self.model._tokenizer.encode("<|im_end|>")[0]

        with torch.no_grad():
            output_ids = self.model._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.model._tokenizer.eos_token_id,
                eos_token_id=eos_token,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.model.decode(new_ids).strip()

        # Strip thinking block
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()
