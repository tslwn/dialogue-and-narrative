"""A demonstrative implementation of the word2vec model."""

import argparse
from collections.abc import Generator, Sequence
from logging import Formatter, Logger, StreamHandler, getLogger
from random import choices

import numpy
import numpy.typing
from numpy.linalg import norm
from nptyping import Float, NDArray, Shape
from Modules.doc2dial import load_dataset

Vector = NDArray[Shape["*"], Float]
Matrix = NDArray[Shape["*, *"], Float]


class Vocabulary:
    """A vocabulary."""

    def __init__(self, alpha: float = 0.75) -> None:
        self.__alpha = alpha

        self.index: dict[str, int] = {}
        self.words: list[str] = []

        self.__counts: list[int] = []
        self.__weights: list[float] = []

    def __len__(self) -> int:
        return len(self.words)

    def fit(self, documents: list[list[str]]) -> None:
        """Fit the vocabulary to the documents."""
        for document in documents:
            for word in document:
                index = self.index.get(word)

                if index is None:
                    index = len(self.words)

                    self.index[word] = index
                    self.words.append(word)

                    self.__counts.append(0)
                    self.__weights.append(0)

                self.__counts[index] += 1

        for index, _ in enumerate(self.words):
            self.__weights[index] = self.__counts[index] ** self.__alpha

    def sample(self, target_index: int) -> int:
        """Sample a word from the vocabulary."""
        while True:
            [index] = choices(range(len(self)), self.__weights)

            if index != target_index:
                return index

    def top(self, n: int) -> list[str]:
        """Return the `n` most common words in the vocabulary."""
        return sorted(
            self.words,
            key=lambda word: self.__counts[self.index[word]],
            reverse=True,
        )[:n]


class TargetContextPairs:
    """Target-context word pairs."""

    def __init__(self, window_size: int) -> None:
        self.__window_size = window_size

        self.target_context_pairs: list[tuple[int, int]] = []

    def fit(self, vocabulary: Vocabulary, documents: list[list[str]]) -> None:
        """Fit the target-context word pairs to the vocabulary and documents."""

        for document in documents:
            for ngram in self.__ngrams(document):
                target_index = self.__window_size // 2

                self.target_context_pairs.extend(
                    [
                        (
                            vocabulary.index[ngram[target_index]],
                            vocabulary.index[context],
                        )
                        for context in ngram[0:target_index]
                        + ngram[target_index + 1 :]
                    ]
                )

    def __ngrams(
        self, document: Sequence[str]
    ) -> Generator[tuple[str, ...], None, None]:
        """Generate n-grams where n - 1 is the window size."""
        n = self.__window_size + 1
        for index, _ in enumerate(document):
            if index + n > len(document):
                break

            yield tuple(document[index : index + n])


class RandomMatrix:
    """
    A two-dimensional matrix of size (x, y) that is initialized with random
    values.
    """

    def __init__(self, x: int, y: int) -> None:
        self.__matrix: NDArray[Shape["*, *"], Float] = numpy.random.rand(x, y)

    def __getitem__(self, index: int) -> Vector:
        """Return the row vector at the given index."""
        return self.__matrix[index]

    def update(self, index: int, value: Vector) -> None:
        """Update the row vector at the given index."""
        self.__matrix[index] += value


def sigmoid(x: float) -> float:
    """The sigmoid function."""
    return 1 / (1 + numpy.exp(-x))


class Word2Vec:
    """A word2vec model."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        logger: Logger = getLogger("word2vec"),
        negative_sample_ratio: int = 2,
        num_dimensions: int = 100,
        num_iterations: int = 10,
        window_size: int = 2,
    ) -> None:
        self.__logger = logger
        self.__learning_rate = learning_rate
        self.__num_dimensions = num_dimensions
        self.__num_iterations = num_iterations
        self.__negative_sample_ratio = negative_sample_ratio
        self.__window_size = window_size

        self.__vocabulary = Vocabulary()
        self.__positive_examples = TargetContextPairs(self.__window_size)

        self.__target_matrix = RandomMatrix(0, self.__num_dimensions)
        self.__context_matrix_positive = RandomMatrix(0, self.__num_dimensions)
        self.__context_matrix_negative = RandomMatrix(0, self.__num_dimensions)

    def fit(self, documents: list[list[str]]) -> None:
        """Fit the model to the documents."""

        self.__vocabulary.fit(documents)
        self.__logger.info("vocabulary size %d", len(self.__vocabulary))

        self.__positive_examples.fit(self.__vocabulary, documents)
        self.__logger.info(
            "positive examples size %d",
            len(self.__positive_examples.target_context_pairs),
        )

        x = len(self.__vocabulary)
        self.__target_matrix = RandomMatrix(x, self.__num_dimensions)
        self.__context_matrix_positive = RandomMatrix(x, self.__num_dimensions)
        self.__context_matrix_negative = RandomMatrix(x, self.__num_dimensions)
        self.__logger.info(
            "matrix size (%d, %d)",
            len(self.__vocabulary),
            self.__num_dimensions,
        )

    def train(self) -> None:
        """Train the model."""

        for index in range(self.__num_iterations):
            self.__logger.info("iteration %d", index + 1)
            self.__train_examples()

    def __train_examples(self) -> None:
        for (
            target_index,
            context_index_positive,
        ) in self.__positive_examples.target_context_pairs:
            self.__train_positive_example(target_index, context_index_positive)

            for context_index_negative in [
                self.__vocabulary.sample(target_index)
                for _ in range(self.__negative_sample_ratio)
            ]:
                self.__train_negative_example(
                    target_index, context_index_negative
                )

    def __train_positive_example(
        self, target_index: int, context_index_positive: int
    ) -> None:
        target_vector = self.__target_matrix[target_index]

        context_vector_positive = self.__context_matrix_positive[
            context_index_positive
        ]

        loss_gradient_context_vector_positive = (
            sigmoid(numpy.dot(target_vector, context_vector_positive)) - 1
        )

        self.__context_matrix_positive.update(
            context_index_positive,
            -(
                self.__learning_rate
                * loss_gradient_context_vector_positive
                * self.__target_matrix[target_index]
            ),
        )

    def __train_negative_example(
        self, target_index: int, context_index_negative: int
    ) -> None:
        target_vector = self.__target_matrix[target_index]

        context_vector_negative = self.__context_matrix_negative[
            context_index_negative
        ]

        loss_gradient_context_vector_negative = sigmoid(
            numpy.dot(
                self.__context_matrix_negative[context_index_negative],
                self.__target_matrix[target_index],
            )
        )

        self.__context_matrix_negative.update(
            context_index_negative,
            -(
                self.__learning_rate
                * loss_gradient_context_vector_negative
                * target_vector
            ),
        )

        self.__target_matrix.update(
            target_index,
            -(
                self.__learning_rate
                * loss_gradient_context_vector_negative
                * context_vector_negative
            ),
        )

    def similarity(self, word1: str, word2: str) -> float:
        """The cosine similarity between two words."""
        vector1 = self.__target_matrix[self.__vocabulary.index[word1]]
        vector2 = self.__target_matrix[self.__vocabulary.index[word2]]

        return numpy.dot(
            vector1,
            vector2,
        ) / (norm(vector1) * norm(vector2))


def create_logger(name: str, level: str) -> Logger:
    """Create a logger."""

    handler = StreamHandler()
    handler.setLevel(level.upper())
    handler.setFormatter(Formatter("%(asctime)s %(name)s %(message)s"))

    logger = getLogger(name)
    logger.setLevel(level.upper())
    logger.addHandler(handler)

    return logger


def get_log_level() -> str:
    """Get the log level from the command-line argument."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l",
        "--log",
        default="warning",
    )

    args = parser.parse_args()

    return args.log


def get_documents(n: int = 100) -> list[list[str]]:
    """Get the documents."""
    return [doc.lower().split(" ") for doc in load_dataset(n)]


if __name__ == "__main__":
    model = Word2Vec(
        logger=create_logger("word2vec", get_log_level()),
        num_dimensions=100,
        num_iterations=10,
    )

    model.fit(get_documents())
    model.train()

    for word3, word4 in [
        ("license", "registration"),
        ("license", "state"),
    ]:
        print(f"{word3} {word4} {model.similarity(word3, word4):.3f}")
