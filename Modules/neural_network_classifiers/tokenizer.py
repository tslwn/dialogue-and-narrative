"""A tokenizer based on scikit-learn's CountVectorizer."""

from collections.abc import Generator, Sequence
from typing import TypedDict
from gensim.utils import tokenize
from sklearn.feature_extraction.text import CountVectorizer
from ..datasets import Document


class DocumentTokens(TypedDict):
    """A document and list of token IDs."""

    text: str
    label: int
    tokens: list[int]


class Tokenizer:
    """A tokenizer based on scikit-learn's CountVectorizer."""

    def __init__(self, documents: Sequence[Document]) -> None:
        self._vectorizer = CountVectorizer(tokenizer=tokenize)

        self._vectorizer.fit([document["text"] for document in documents])  # type: ignore

        self.vocabulary: dict[str, int] = self._vectorizer.vocabulary_  # type: ignore

    def map(
        self, documents: Sequence[Document]
    ) -> Generator[DocumentTokens, None, None]:
        """
        Transform a list of documents into a list of documents with lists of
        token IDs.
        """

        for document in documents:
            yield self.map_one(document)

    def map_one(self, document: Document) -> DocumentTokens:
        """Transform a document into a document with a list of token IDs."""

        tokens: list[int] = []

        for token in tokenize(document["text"]):
            # Skip tokens that are not in the vocabulary.
            if token in self.vocabulary:
                # Reserve zero to pad sequences.
                tokens.append(self.vocabulary[token] + 1)

        return DocumentTokens(
            text=document["text"],
            label=document["label"],
            tokens=tokens,
        )

    def pad(
        self, documents: list[DocumentTokens], length: int
    ) -> list[DocumentTokens]:
        """
        Pad or truncate a list of documents with lists of token IDs to a given
        length.
        """

        return [self.pad_one(document, length) for document in documents]

    def pad_one(self, document: DocumentTokens, length: int) -> DocumentTokens:
        """Pad or truncate a document's list of token IDs to a given length."""

        if len(document["tokens"]) >= length:
            tokens = document["tokens"][:length]

        else:
            zeros = length - len(document["tokens"])
            tokens = [0] * zeros + document["tokens"]

        return DocumentTokens(
            text=document["text"], label=document["label"], tokens=tokens
        )
