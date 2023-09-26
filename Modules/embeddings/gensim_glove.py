"""A wrapper around gensim's `glove`."""

from collections.abc import Generator
from gensim.downloader import load
from gensim.models import KeyedVectors
from numpy import float64
from numpy.typing import NDArray
from .abstract_term_vectors import AbstractTermVectors


class GensimGlove(AbstractTermVectors):
    """A wrapper around gensim's `glove`."""

    def __init__(self, _documents: list[str], name: str = "glove-twitter-25"):
        super().__init__(_documents)
        self.vectors: KeyedVectors = load(name)

    def terms(self) -> Generator[str, None, None]:
        for term in self.vectors.index_to_key:
            yield term

    def term_index(self, term: str) -> int:
        return self.vectors.key_to_index[term]

    def term_vector(self, term: str) -> NDArray[float64]:
        return self.vectors[term]
