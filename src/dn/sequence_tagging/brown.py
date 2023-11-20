"""A wrapper around the Brown tagged-sentences dataset."""

from logging import Logger
from typing import Any, Generic
import nltk
import nltk.corpus
from nltk.corpus.reader.util import ConcatenatedCorpusView
from numpy import int64, ndarray, str_
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ..log import create_logger
from .util import Arr, Int, flat_nested_list, map_nested_list, pickler


class BrownTaggedSentences(Generic[Int]):
    """A wrapper around the Brown tagged-sentences dataset."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        logger: Logger = create_logger("brown", "info"),
    ):
        self.__test_size = test_size
        self.__random_state = random_state
        self.__logger = logger

        self.__word_encoder = LabelEncoder()
        self.__tag_encoder = LabelEncoder()

        logger.info("downloading brown")
        nltk.download("brown", quiet=True)
        logger.info("downloading universal_tagset")
        nltk.download("universal_tagset", quiet=True)

        tagged_sentences = nltk.corpus.brown.tagged_sents(tagset="universal")
        assert isinstance(tagged_sentences, ConcatenatedCorpusView)
        self.__logger.info("%d sentences", len(tagged_sentences))

        tagged_sentences_train, tagged_sentences_test = train_test_split(
            tagged_sentences,
            test_size=self.__test_size,
            random_state=self.__random_state,
        )

        del tagged_sentences

        words_train, tags_train = self.__split_tagged_sentences(
            tagged_sentences_train
        )
        words_test, tags_test = self.__split_tagged_sentences(
            tagged_sentences_test
        )

        del tagged_sentences_train, tagged_sentences_test

        (
            self.words_train_encoded,
            self.words_test_encoded,
        ) = self.__encode_split(self.__word_encoder, words_train, words_test)
        (
            self.tags_train_encoded,
            self.tags_test_encoded,
        ) = self.__encode_split(self.__tag_encoder, tags_train, tags_test)

        self.n_words = len(self.__word_encoder.classes_)
        self.__logger.info("%d words", self.n_words)
        self.n_tags = len(self.__tag_encoder.classes_)
        self.__logger.info("%d tags", self.n_tags)

    def decode_words(self, words: list[Int]) -> Arr[str_]:
        """Decode a list of words."""
        return self.__word_encoder.inverse_transform(words)

    def decode_tags(self, tags: list[Int]) -> Arr[str_]:
        """Decode a list of tags."""
        return self.__tag_encoder.inverse_transform(tags)

    def __encode_split(
        self,
        encoder: LabelEncoder,
        train: list[list[str]],
        test: list[list[str]],
    ):
        train_flat = flat_nested_list(train)
        test_flat = flat_nested_list(test)

        encoder.fit(train_flat + test_flat)

        _train_flat_encoded = encoder.transform(train_flat)
        _test_flat_encoded = encoder.transform(test_flat)

        assert isinstance(_train_flat_encoded, ndarray)
        assert isinstance(_test_flat_encoded, ndarray)

        train_flat_encoded: Arr[Int] = _train_flat_encoded
        test_flat_encoded: Arr[Int] = _test_flat_encoded

        train_encoded = map_nested_list(train, train_flat_encoded)
        test_encoded = map_nested_list(test, test_flat_encoded)

        return train_encoded, test_encoded

    def __split_tagged_sentences(
        self,
        tagged_sentences: Any,
    ) -> tuple[list[list[str]], list[list[str]]]:
        assert isinstance(tagged_sentences, list)
        sentences_words: list[list[str]] = []
        sentences_tags: list[list[str]] = []
        for tagged_sentence in tagged_sentences:
            sentence_words: list[str] = []
            sentence_tags: list[str] = []
            assert isinstance(tagged_sentence, list)
            for tagged_word in tagged_sentence:
                assert isinstance(tagged_word, tuple)
                assert len(tagged_word) == 2
                assert isinstance(tagged_word[0], str)
                assert isinstance(tagged_word[1], str)
                sentence_words.append(tagged_word[0])
                sentence_tags.append(tagged_word[1])
            sentences_words.append(sentence_words)
            sentences_tags.append(sentence_tags)
        return sentences_words, sentences_tags


@pickler("brown_tagged_sentences.pickle")
def get_brown_tagged_sentences() -> BrownTaggedSentences[int64]:
    """Get the Brown tagged-sentences dataset."""
    return BrownTaggedSentences()
