import os
from llm_sdk import Small_LLM_Model
from src.models import MinimalSource


SYSTEM_PROMPT = """You are a precise technical assistant for the vLLM codebase.
Answer ONLY using the context provided below.
Do NOT use prior knowledge or make assumptions beyond what the context states.
If the answer is not present in the context, respond with: "Not found in the
provided sources."
Keep your answer concise, self-contained, and cite the source file(s) you
used."""


def _build_prompt(question: str, context_blocks: list[str]) -> str:
    """Assemble the full prompt fed to the LLM.

    Structure:
        [SYSTEM]
        CONTEXT:
        --- file: path ---
        chunk text
        ...
        QUESTION: ...
        ANSWER:

    The trailing 'ANSWER:' token primes the model to generate a response
    directly, reducing preamble like "Sure, I can help you with that...".

    Args:
        question: The user's natural-language question.
        context_blocks: List of formatted context strings (one per chunk).

    Returns:
        Full prompt string ready for tokenization.
    """
    context_section = "\n\n".join(context_blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_section}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )


class Generator:
    """Loads Qwen/Qwen3-0.6B and generates grounded answers via greedy
    decoding.

    Greedy decoding: at each step, pick the token with the highest logit.
    This is deterministic and avoids hallucination drift from sampling.

    The model is loaded once at construction and reused for all calls,
    which is why Generator should be instantiated once and shared
    (e.g. via the Pipeline class).

    Usage:
        generator = Generator()
        answer = generator.answer("How does PagedAttention work?", sources)

    Attributes:
        model: Loaded Small_LLM_Model instance.
        max_new_tokens: Maximum tokens to generate per answer.
        repo_root: Used to resolve file paths when reading chunk content.
    """

    MODEL_NAME = "Qwen/Qwen3-0.6B"

    def __init__(self,
                 max_new_tokens: int = 150,
                 repo_root: str = "data/raw/vllm-0.10.1") -> None:
        """Load the LLM. This is the slow step (~30-60s on first run).

        Args:
            max_new_tokens: Maximum tokens generated per answer.
                Keep low (100-200) — greedy decoding has no KV-cache,
                so each token costs a full forward pass.
            repo_root: Root of the vLLM repo, used to read chunk text.
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

        context_blocks = self._build_context_blocks(sources)
        prompt = _build_prompt(question, context_blocks)
        prompt = self._truncate_prompt(prompt)
        return self._greedy_decode(prompt)

    def answer_batch(self, questions: list[str],
                     sources_list: list[list[MinimalSource]]
                     ) -> list[str]:
        """Generate answers for multiple questions.

        Args:
            questions: List of question strings.
            sources_list: Parallel list of retrieved sources per question.

        Returns:
            List of answer strings, same order as input.
        """
        answers: list[str] = []
        for question, sources in zip(questions, sources_list):
            try:
                answers.append(self.answer(question, sources))
            except Exception:
                answers.append("Error generating answer.")
        return answers

    def _build_context_blocks(self,
                              sources: list[MinimalSource]) -> list[str]:
        """Read the actual text for each source from disk and format it.

        Args:
            sources: Retrieved source objects with file_path and char indices.

        Returns:
            List of formatted strings: '--- file: path ---\\nchunk text'.
        """
        blocks: list[str] = []
        for source in sources:
            text = self._read_chunk_text(source)
            if text:
                block = f"--- file: {source.file_path} ---\n{text}"
                blocks.append(block)
        return blocks

    def _read_chunk_text(self, source: MinimalSource) -> str:
        """Extract the exact text slice from the source file on disk.

        Args:
            source: MinimalSource with file_path and character indices.

        Returns:
            The text slice, or empty string if the file cannot be read.
        """
        abs_path = os.path.join(self.repo_root, source.file_path)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            return content[source.first_character_index:source.last_character_index]
        except (OSError, IndexError):
            return ""

    def _truncate_prompt(self, prompt: str, max_chars: int = 3000) -> str:
        """Truncate the prompt to stay within the model's context window.

        Qwen3-0.6B has a 32k token context, but with greedy decoding
        (no KV-cache) longer prompts are proportionally slower.
        We keep the system prompt and question intact, truncating only
        the middle context section.

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

    def _greedy_decode(self, prompt: str) -> str:
        """Generate text token by token using argmax (greedy) decoding.

        This is the only generation strategy available through the SDK's
        public interface (get_logits_from_input_ids).

        Each iteration:
          1. Tokenize the current sequence.
          2. Run a forward pass → get logits for the next token position.
          3. Pick the token with the highest logit (argmax).
          4. Append it and repeat.
          5. Stop at EOS or max_new_tokens.

        Args:
            prompt: The full prompt string to continue from.

        Returns:
            Generated text (new tokens only, not including the prompt).
        """
        input_ids: list[int] = self.model.encode(prompt)[0].tolist()
        prompt_length = len(input_ids)
        eos_id: int = self.model._tokenizer.eos_token_id

        for _ in range(self.max_new_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_token = int(max(range(len(logits)), key=lambda i: logits[i]))

            if next_token == eos_id:
                break

            input_ids.append(next_token)

        new_ids = input_ids[prompt_length:]
        return self.model.decode(new_ids).strip()
