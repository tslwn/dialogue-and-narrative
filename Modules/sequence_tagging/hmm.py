"""A Hidden Markov Model tagger."""

from logging import Logger
from typing import Any, Generic
import numpy
import numpy.typing
from ..log import create_logger
from .brown import get_brown_tagged_sentences
from .util import Int

EPSILON = 1e-7

hmm_logger = create_logger("hmm", "info")


class HMMTagger(Generic[Int]):
    """A Hidden Markov Model tagger."""

    def __init__(
        self,
        n_words: int,
        n_tags: int,
        logger: Logger = hmm_logger,
    ):
        self.__n_words = n_words
        self.__n_tags = n_tags
        self.__logger = logger

        self.__initial_matrix: numpy.ndarray[
            Any, numpy.dtype[numpy.float_]
        ] | None = None
        self.__transition_matrix: numpy.ndarray[
            Any, numpy.dtype[numpy.float_]
        ] | None = None
        self.__emission_matrix: numpy.ndarray[
            Any, numpy.dtype[numpy.float_]
        ] | None = None

    def fit(
        self,
        train_words: list[list[Int]],
        train_tags: list[list[Int]],
    ):
        """Fit the model."""
        self.__initial_matrix = self._initial_matrix(train_tags)
        self.__logger.info("__initial_matrix %s", self.__initial_matrix.shape)

        self.__transition_matrix = self._transition_matrix(
            train_words, train_tags
        )
        self.__logger.info(
            "__transition_matrix %s", self.__transition_matrix.shape
        )

        self.__emission_matrix = self._emission_matrix(train_words, train_tags)
        self.__logger.info("__emission_matrix %s", self.__emission_matrix.shape)

    def _initial_matrix(self, train_tags: list[list[Int]]):
        initial_matrix = numpy.zeros(self.__n_tags)
        for tags in train_tags:
            initial_matrix[tags[0]] += 1
        initial_matrix /= len(train_tags)
        return initial_matrix

    def _transition_matrix(
        self,
        train_words: list[list[Int]],
        train_tags: list[list[Int]],
    ):
        transition_matrix = numpy.zeros((self.__n_tags, self.__n_tags))
        for words, tags in zip(train_words, train_tags):
            for index in range(len(words) - 1):
                tag_1 = tags[index]
                tag_2 = tags[index + 1]
                transition_matrix[tag_1, tag_2] += 1
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
        return transition_matrix

    def _emission_matrix(
        self,
        train_words: list[list[Int]],
        train_tags: list[list[Int]],
    ):
        emission_matrix = numpy.zeros((self.__n_tags, self.__n_words))
        for words, tags in zip(train_words, train_tags):
            for word, tag in zip(words, tags):
                emission_matrix[tag, word] += 1
        emission_matrix /= emission_matrix.sum(axis=1, keepdims=True)
        return emission_matrix

    def predict(
        self,
        words: list[Int],
    ) -> numpy.ndarray[Any, numpy.dtype[Int]]:
        """Find the most-likely sequence of tags."""
        assert self.__initial_matrix is not None
        assert self.__transition_matrix is not None
        assert self.__emission_matrix is not None

        n_words = len(words)

        # Initialise the Viterbi and backpointer matrices.
        viterbi_matrix = numpy.zeros((n_words, self.__n_tags))
        backpointer_matrix = numpy.zeros((n_words, self.__n_tags))

        # Compute the probabilities of the initial states.
        viterbi_matrix[0, :] = (
            self.__initial_matrix * self.__emission_matrix[:, words[0]]
        )

        # For each observation (word)...
        for word_index in range(1, n_words):
            # For each possible state (tag)...
            for tag_index in range(self.__n_tags):
                # Compute the probabilities of the possible previous states.
                previous_tag_probabilities = (
                    viterbi_matrix[word_index - 1, :]
                    * self.__transition_matrix[:, tag_index]
                )

                # Find the most-likely previous state.
                max_previous_tag_probability = numpy.max(
                    previous_tag_probabilities
                )
                max_previous_tag = numpy.argmax(previous_tag_probabilities)

                # Compute the probability of the most-likely previous state.
                viterbi_matrix[word_index, tag_index] = (
                    max_previous_tag_probability + EPSILON
                ) * (
                    self.__emission_matrix[tag_index, words[word_index]]
                    + EPSILON
                )

                # Store the most-likely previous state.
                backpointer_matrix[word_index, tag_index] = max_previous_tag

        word_index = n_words - 1

        # Initialise the final state.
        tags = numpy.zeros(n_words, dtype=int)

        # Find the most-likely final state.
        tags[word_index] = numpy.argmax(viterbi_matrix[word_index, :])

        # Backtrack to the first observation.
        for word_index in range(len(words) - 1, 0, -1):
            tags[word_index - 1] = backpointer_matrix[
                word_index, tags[word_index]
            ]

        return tags


if __name__ == "__main__":
    brown_tagged_sentences = get_brown_tagged_sentences()

    hmm_tagger = HMMTagger[numpy.int64](
        brown_tagged_sentences.n_words, brown_tagged_sentences.n_tags
    )

    hmm_tagger.fit(
        brown_tagged_sentences.words_train_encoded,
        brown_tagged_sentences.tags_train_encoded,
    )

    # Print the predicted tags for the first N sentences.
    N_SENTENCES = 5
    for test_words, test_tags in zip(
        brown_tagged_sentences.words_test_encoded[:N_SENTENCES],
        brown_tagged_sentences.tags_test_encoded[:N_SENTENCES],
    ):
        pred_tags = list(
            hmm_tagger.predict(
                test_words,
            )
        )

        test_words_decoded = brown_tagged_sentences.decode_words(test_words)
        test_tags_decoded = brown_tagged_sentences.decode_tags(test_tags)
        pred_tags_decoded = brown_tagged_sentences.decode_tags(pred_tags)

        hmm_logger.info(" ".join(test_words_decoded))
        hmm_logger.info(" ".join(test_tags_decoded))
        hmm_logger.info(" ".join(pred_tags_decoded))

    # Compute the accuracy for the test set.
    correct: int = 0
    for test_words, test_tags in zip(
        brown_tagged_sentences.words_test_encoded,
        brown_tagged_sentences.tags_test_encoded,
    ):
        pred_tags = list(
            hmm_tagger.predict(
                test_words,
            )
        )
        correct += numpy.sum(pred_tags == test_tags)

    accuracy = correct / len(brown_tagged_sentences.tags_test_encoded)
    hmm_logger.info("accuracy %s", accuracy)
