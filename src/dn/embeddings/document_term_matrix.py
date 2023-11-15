"""Utilities to build a document-term matrix."""

from collections.abc import Generator
from numpy import float64, str_
from numpy.typing import NDArray
from sklearn.feature_extraction.text import CountVectorizer
from .abstract_term_vectors import AbstractTermVectors


class DocumentTermMatrix(AbstractTermVectors):
    """
    A document-term matrix based on scikit-learn's `CountVectorizer`.
    """

    def __init__(
        self, documents: list[str], ngram_range: tuple[int, int] = (1, 1)
    ):
        super().__init__(documents)
        self.__vectorizer = CountVectorizer(ngram_range=ngram_range)
        self.__vectorizer.fit(documents)
        self.matrix = self.__vectorizer.transform(documents)
        self.array = self.matrix.toarray()

    def get_feature_names(self) -> NDArray[str_]:
        """Get the feature names."""
        return self.__vectorizer.get_feature_names_out()

    def transform(self, documents: list[str]) -> NDArray[float64]:
        """Transform a list of documents into a document-term matrix."""
        return self.__vectorizer.transform(documents).toarray()

    def terms(self) -> Generator[str, None, None]:
        for term in self.__vectorizer.vocabulary_.keys():
            yield term

    def term_index(self, term: str) -> int:
        return self.__vectorizer.vocabulary_[term]

    def term_vector(self, term: str) -> NDArray[float64]:
        return self.matrix.getcol(self.term_index(term)).toarray().flatten()

    def term_documents(
        self, term: str, n: int = 10
    ) -> Generator[str, None, None]:
        """Iterate the documents that include a term."""
        count = 0
        for document_index, document_count in enumerate(self.term_vector(term)):
            if count >= n:
                return
            if document_count > 0:
                count += 1
                yield self.documents[document_index]

    def document_vector(self, document_index: int):
        """Get the vector for a document."""
        return self.matrix.getrow(document_index).toarray().flatten()

    def document_terms(self, document_index: int):
        """Get the terms for a document."""

        for term_index, term_count in enumerate(
            self.document_vector(document_index)
        ):
            if term_count > 0:
                yield self.get_feature_names()[term_index]
