"""A wrapper around gensim's `word2vec`."""

from collections.abc import Generator
from gensim.models.word2vec import Word2Vec
from gensim.utils import tokenize
from numpy import float64
from numpy.typing import NDArray
from .abstract_term_vectors import AbstractTermVectors


class GensimWord2Vec(AbstractTermVectors):
    """A wrapper around gensim's `word2vec`."""

    def __init__(self, documents: list[str]):
        super().__init__(documents)
        self.__word2vec = Word2Vec(
            list(
                list(tokenize(document, lowercase=True))
                for document in documents
            ),
            sg=1,
            min_count=1,
            window=3,
            vector_size=25,
        )

    def terms(self) -> Generator[str, None, None]:
        for term in self.__word2vec.wv.index_to_key:
            yield term

    def term_index(self, term: str) -> int:
        return self.__word2vec.wv.key_to_index[term]

    def term_vector(self, term: str) -> NDArray[float64]:
        return self.__word2vec.wv[term]
