import re
import torch
from llm_sdk.__init__ import Small_LLM_Model

NOT_FOUND_PATTERNS = (
    r"not found in the provided sources",
    r"i've not found",
    r"i have not found",
    r"the provided sources do not contain",
    r"there is no information",
    r"no relevant information",
)

NOT_FOUND_MESSAGE = "Not found in the provided sources."


def build_prompt(question: str, context_blocks: list[str]) -> str:
    """Assemble prompt using Qwen3 chat format with thinking disabled.

    Args:
        question: The natural-language question.
        context_blocks: Pre-built context strings, one per source.

    Returns:
        Full chat-formatted prompt string.
    """
    context_section = "\n\n".join(context_blocks)
    system = (
        "You are a technical assistant for the vLLM codebase. "
        "Answer using ONLY the CONTEXT provided. No prior knowledge.\n\n"
        "RULES:\n"
        "1. If the answer is in the CONTEXT: give a direct answer and "
        "end with 'Source: <filename>'.\n"
        "2. If the answer is NOT in the CONTEXT: respond with exactly "
        "'Not found in the provided sources.' Nothing else.\n"
        "3. Do not infer, guess, or add information not in the "
        "CONTEXT.\n\n"
        "EXAMPLES:\n"
        "Q: What does X do?\n"
        "A: X does Y. Source: vllm/x.py\n\n"
        "Q: What is the capital of France?\n"
        "A: Not found in the provided sources.\n"
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
    """Loads Qwen/Qwen3-0.6B and generates grounded answers via
    greedy decoding.

    Greedy decoding picks the highest-logit token at each step,
    making generation deterministic and faithful to the retrieved
    context.

    The model is loaded once at construction and reused for all
    calls. Instantiate once and share via the CLI lazy loader.

    Usage:
        generator = Generator()
        answer = generator.answer_with_text(
            "How does PagedAttention work?", context_blocks
        )

    Attributes:
        model: Loaded Small_LLM_Model instance.
        max_new_tokens: Maximum tokens to generate per answer.
        repo_root: Root of the vLLM repo (kept for compatibility).
    """

    MODEL_NAME = "Qwen/Qwen3-0.6B"

    def __init__(self, max_new_tokens: int = 150,
                 repo_root: str = "data/raw/vllm-0.10.1") -> None:
        """Load the LLM. First run downloads weights (~600 MB).

        Args:
            max_new_tokens: Cap on generated tokens per answer. Keep
                low (100-200) — each token costs a full forward pass.
            repo_root: Root of the vLLM repo, kept for backward
                compatibility with callers that pass it.
        """
        print(f"Loading {self.MODEL_NAME}...")
        self.model = Small_LLM_Model(model_name=self.MODEL_NAME)
        self.max_new_tokens = max_new_tokens
        self.repo_root = repo_root
        print("Model loaded.")

    def answer_with_text(self, question: str,
                         context_blocks: list[str]) -> str:
        """Generate a grounded answer from pre-built context blocks.

        Args:
            question: The natural-language question.
            context_blocks: Pre-built context strings, one per
                source, typically built from parent chunks already
                held in memory by the Retriever.

        Returns:
            A concise, source-grounded answer string.
        """
        if not question or not question.strip():
            return "No question provided."
        if not context_blocks:
            return "No relevant sources found."

        prompt = build_prompt(question, context_blocks)
        prompt = self.truncate_prompt(prompt)
        return self.greedy_decode(prompt)

    def truncate_prompt(self, prompt: str,
                        max_chars: int = 10000) -> str:
        """Truncate the prompt to stay within a reasonable token
        budget.

        Keeps the system prompt and question intact; truncates only
        the middle context section so the model always sees the
        full question.

        Args:
            prompt: Full assembled prompt.
            max_chars: Approximate character budget (1 token ~= 4
                chars).

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
        """Generate text using HuggingFace's generate() with
        KV-cache.

        Strips the model's thinking block, then normalises any
        "no answer" phrasing to a single canonical message so
        downstream callers can match on it reliably.

        Args:
            prompt: The full prompt string to continue from.

        Returns:
            Generated text (new tokens only, prompt excluded).
        """
        inputs = self.model._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
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

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.strip()

        text_lower = text.lower()
        for pattern in NOT_FOUND_PATTERNS:
            if re.search(pattern, text_lower):
                return NOT_FOUND_MESSAGE

        return text