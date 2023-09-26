"""Solutions to the exercises in chapter 4."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import log

ClassName = str
DocumentIndex = int
Token = str
TokenCount = int
TokenIndex = int


class Document:
    """A document."""

    def __init__(self, tokens: list[Token], class_name: ClassName) -> None:
        self.tokens = tokens
        self.class_name = class_name


class Vocab:
    """A vocabulary."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.tokens: list[str] = []
        self.index: dict[str, int] = {}

        if tokens is not None:
            for token in tokens:
                self.add(token)

    def add(self, token: str) -> int:
        """Add a token to the vocabulary."""
        index = self.index.get(token)
        if index is None:
            index = len(self.tokens)
            self.tokens.append(token)
            self.index[token] = index
        return index


class AbstractNaiveBayes:
    """An abstract naive Bayes classifier."""

    def __init__(
        self,
        vocab: Vocab | None = None,
        log_prior: dict[ClassName, float] | None = None,
        log_likelihood: dict[ClassName, dict[TokenIndex, float]] | None = None,
    ) -> None:
        self._vocab = vocab or Vocab()
        self._log_prior = log_prior or {}
        self._log_likelihood = log_likelihood or {}

    def predict(self, tokens: Sequence[str]) -> ClassName:
        """Predict the class of the tokens."""
        log_likelihood_by_class: dict[ClassName, float] = {}
        for class_name in self._log_prior:
            log_likelihood_by_class[class_name] = self._log_prior[class_name]
            for token in tokens:
                token_index = self._vocab.index.get(token)
                if token_index is not None:
                    log_likelihood_by_class[class_name] += self._log_likelihood[
                        class_name
                    ][token_index]
        return max(
            log_likelihood_by_class,
            key=lambda class_name: log_likelihood_by_class[class_name],
        )


class AbstractNaiveBayesText(AbstractNaiveBayes, ABC):
    """An abstract naive Bayes text classifier."""

    def __init__(self, k: float = 0.0) -> None:
        super().__init__()
        self._k = k
        self._documents: list[Document] = []
        self._documents_by_class: dict[ClassName, list[DocumentIndex]] = {}
        self._features_by_class: dict[ClassName, dict[TokenIndex, int]] = {}

    @abstractmethod
    def _class_features(self, documents: list[Document]) -> None:
        """Compute the token features in each class."""

    def fit(self, documents: list[Document]) -> None:
        """Fit the classifier to the documents."""
        for document in documents:
            document_index = len(self._documents)
            self._documents.append(document)
            self._documents_by_class.setdefault(document.class_name, []).append(
                document_index
            )
            for token in document.tokens:
                self._vocab.add(token)

        self._class_features(documents)

        # Compute the log prior probabilities and log likelihoods.
        for class_name, document_indexes in self._documents_by_class.items():
            self._log_prior[class_name] = log(
                len(document_indexes) / len(self._documents)
            )

            class_features = self._features_by_class[class_name]
            class_token_count = sum(class_features.values())

            self._log_likelihood[class_name] = {}
            for token in self._vocab.tokens:
                token_index = self._vocab.index[token]
                self._log_likelihood[class_name][token_index] = log(
                    (class_features.get(token_index, 0) + self._k)
                    / (class_token_count + self._k * len(self._vocab.tokens))
                )


class NaiveBayesText(AbstractNaiveBayesText):
    """A naive Bayes text classifier."""

    def __init__(self, k: float = 0.0) -> None:
        super().__init__(k=k)

    def _class_features(self, documents: list[Document]) -> None:
        """Compute the number of times each token appears in each class."""
        for document in documents:
            for token in document.tokens:
                token_index = self._vocab.index.get(token)
                if token_index is not None:
                    class_features = self._features_by_class.setdefault(
                        document.class_name, {}
                    )
                    class_features[token_index] = (
                        class_features.get(token_index, 0) + 1
                    )


class NaiveBayesTextBinarized(AbstractNaiveBayesText):
    """A binarized naive Bayes text classifier."""

    def __init__(self, k: float = 0.0) -> None:
        super().__init__(k=k)

    def _class_features(self, documents: list[Document]) -> None:
        """Compute the number of documents in which each token appears in each class."""
        for token in self._vocab.tokens:
            token_index = self._vocab.index[token]
            for document in documents:
                if token in document.tokens:
                    class_features = self._features_by_class.setdefault(
                        document.class_name, {}
                    )
                    class_features[token_index] = (
                        class_features.get(token_index, 0) + 1
                    )


def exercise_4_1():
    """Exercise 4.1."""

    def log_likelihood(likelihoods: list[float]) -> dict[TokenIndex, float]:
        return {
            index: log(likelihood)
            for index, likelihood in enumerate(likelihoods)
        }

    naive_bayes = AbstractNaiveBayes(
        vocab=Vocab(tokens=["I", "always", "like", "foreign", "films"]),
        log_prior={
            "pos": log(0.5),
            "neg": log(0.5),
        },
        log_likelihood={
            "pos": log_likelihood([0.09, 0.07, 0.29, 0.04, 0.08]),
            "neg": log_likelihood([0.16, 0.06, 0.06, 0.15, 0.11]),
        },
    )

    print(
        f"Exercise 4.1: {naive_bayes.predict(['I', 'always', 'like', 'foreign', 'films'])}"
    )


def exercise_4_2():
    """Exercise 4.2."""
    documents = [
        Document(["fun", "couple", "love", "love"], "comedy"),
        Document(["fast", "furious", "shoot"], "action"),
        Document(["couple", "fly", "fast", "fun", "fun"], "comedy"),
        Document(["furious", "shoot", "shoot", "fun"], "action"),
        Document(["fly", "fast", "shoot", "love"], "action"),
    ]

    naive_bayes = NaiveBayesText(k=1.0)
    naive_bayes.fit(documents)

    print(f"Exercise 4.2: {naive_bayes.predict(['fast', 'couple', 'shoot'])}")


def exercise_4_3():
    """Exercise 4.3."""
    documents = [
        Document(["good", "good", "good", "great", "great", "great"], "pos"),
        Document(["poor", "great", "great"], "pos"),
        Document(["good", "poor", "poor", "poor"], "neg"),
        Document(
            ["good", "poor", "poor", "poor", "poor", "poor", "great", "great"],
            "neg",
        ),
        Document(["poor", "poor"], "neg"),
    ]

    tokens = "A good good plot and great characters but poor acting".split(" ")

    naive_bayes = NaiveBayesText(k=1.0)
    naive_bayes.fit(documents)

    print(f"Exercise 4.3 (multinomial): {naive_bayes.predict(tokens)}")

    naive_bayes_binarized = NaiveBayesTextBinarized(k=1.0)
    naive_bayes_binarized.fit(documents)

    print(f"Exercise 4.3 (binarized): {naive_bayes_binarized.predict(tokens)}")


if __name__ == "__main__":
    exercise_4_1()
    exercise_4_2()
    exercise_4_3()
