"""An abstract class wrapper to work with term-vector models."""

from abc import abstractmethod
from collections.abc import Generator
from typing import Any
from numpy import ndarray
from ..maths import similarity


class AbstractTermVectors:
    """
    An abstract class wrapper to work with term-vector models.
    """

    def __init__(self, documents: list[str]):
        self._documents = documents

    @abstractmethod
    def terms(self) -> Generator[str, None, None]:
        """Iterate the terms."""

    @abstractmethod
    def term_index(self, term: str) -> int:
        """Get the index of a term."""

    @abstractmethod
    def term_vector(self, term: str) -> ndarray[Any, Any]:
        """Get the vector for a term."""

    def term_similarity(self, term1: str, term2: str) -> float:
        """Get the similarity between two terms."""
        return similarity(self.term_vector(term1), self.term_vector(term2))

    def most_similar(self, term1: str, n: int = 10) -> list[tuple[str, float]]:
        """Get the `n` most similar terms to a term."""
        return sorted(
            [
                (term2, self.term_similarity(term1, term2))
                for term2 in self.terms()
                if term2 != term1
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:n]
