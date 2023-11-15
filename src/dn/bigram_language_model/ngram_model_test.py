"""Tests for N-gram models with optional add-k smoothing."""

# pylint: disable=missing-class-docstring, missing-function-docstring
# pyright: reportUnknownMemberType=false

import pytest
from .ngram_model import NgramModel


docs = [
    ["foo", "bar", "baz"],
    ["foo", "bar", "qux"],
    ["quux", "bar", "baz"],
]


class TestBigramModel:
    model = NgramModel(order=2)
    model.fit(docs)

    def test_vocab(self):
        assert self.model.vocab == {
            "<s>": 3,
            "foo": 2,
            "bar": 3,
            "baz": 2,
            "</s>": 3,
            "qux": 1,
            "quux": 1,
        }

    def test_ngrams(self):
        assert self.model.counts == {
            ("</s>", "<s>"): 2,
            ("<s>", "foo"): 2,
            ("<s>", "quux"): 1,
            ("bar", "baz"): 2,
            ("bar", "qux"): 1,
            ("baz", "</s>"): 2,
            ("foo", "bar"): 2,
            ("quux", "bar"): 1,
            ("qux", "</s>"): 1,
        }

    def test_score(self):
        assert self.model.score("foo") == pytest.approx(2 / 15)

    def test_score_context(self):
        assert self.model.score("foo", ("<s>",)) == pytest.approx(2 / 3)

    def test_logscore(self):
        assert self.model.logscore("foo") == pytest.approx(-2.906891)

    def test_logscore_context(self):
        assert self.model.logscore("foo", ("<s>",)) == pytest.approx(-0.584962)


class TestTrigramModel:
    model = NgramModel(order=3)
    model.fit(docs)

    def test_ngrams(self):
        assert self.model.counts == {
            ("</s>", "<s>", "foo"): 1,
            ("</s>", "<s>", "quux"): 1,
            ("<s>", "foo", "bar"): 2,
            ("<s>", "quux", "bar"): 1,
            ("bar", "baz", "</s>"): 2,
            ("bar", "qux", "</s>"): 1,
            ("baz", "</s>", "<s>"): 1,
            ("foo", "bar", "baz"): 1,
            ("foo", "bar", "qux"): 1,
            ("quux", "bar", "baz"): 1,
            ("qux", "</s>", "<s>"): 1,
        }

    def test_score_context(self):
        assert self.model.score("baz", ("foo", "bar")) == pytest.approx(1 / 2)

    def test_logscore_context(self):
        assert self.model.logscore("baz", ("foo", "bar")) == pytest.approx(-1.0)


class TestLaplaceModel:
    model = NgramModel(order=2, k=1.0)
    model.fit(docs)

    def test_score(self):
        assert self.model.score("foo") == pytest.approx(3 / 22)

    def test_score_context(self):
        print(self.model.context_counts)
        print(self.model.counts)
        assert self.model.score("foo", ("<s>",)) == pytest.approx(3 / 10)

    def test_logscore(self):
        assert self.model.logscore("foo") == pytest.approx(-2.874469)

    def test_logscore_context(self):
        assert self.model.logscore("foo", ("<s>",)) == pytest.approx(-1.736966)
