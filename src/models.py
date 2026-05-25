import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A chunk of a file retrieved from the knowledge base.

    Attributes:
        file_path: Relative path to the source file inside the repository.
        first_character_index: Start position of the chunk in the file
        (inclusive).
        last_character_index: End position of the chunk in the file
        (exclusive).
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without a ground-truth answer.

    Attributes:
        question_id: Unique identifier, auto-generated if not provided.
        question: The natural-language question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth answer and source locations.

    Attributes:
        sources: List of file chunks that contain the answer.
        answer: Ground-truth answer string.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions (answered or unanswered).

    Attributes:
        rag_questions: List of questions, may be answered or unanswered.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search results for a single question.

    Attributes:
        question_id: Matches the source question's identifier.
        question: The original question text.
        retrieved_sources: Ranked list of retrieved chunks.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results for a single question plus the generated answer.

    Attributes:
        answer: LLM-generated answer grounded in retrieved_sources.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Output of the search_dataset command (subject-spec name).

    Attributes:
        search_results: One MinimalSearchResults entry per question.
        k: Number of chunks retrieved per question.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of the answer_dataset command (subject-spec name).

    Attributes:
        search_results: One MinimalAnswer entry per question.
        k: Number of chunks retrieved per question.
    """

    search_results: List[MinimalAnswer]
    k: int
