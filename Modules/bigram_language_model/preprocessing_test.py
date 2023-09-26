"""Tests for preprocessing utilities."""

# pylint: disable=missing-class-docstring, missing-function-docstring

from .preprocessing import (
    ngrams,
    pad,
    tokenize,
    train_test_split,
)


def test_tokenize():
    assert tokenize("foo bar baz qux quux") == [
        "foo",
        "bar",
        "baz",
        "qux",
        "quux",
    ]


class TestPad:
    def test_default(self):
        assert list(pad(tokenize("foo bar baz qux quux"))) == [
            "<s>",
            "foo",
            "bar",
            "baz",
            "qux",
            "quux",
            "</s>",
        ]

    def test_pad_symbols(self):
        assert list(
            pad(
                tokenize("foo bar baz qux quux"),
                left_pad_symbol="<quuz>",
                right_pad_symbol="<corge>",
            )
        ) == [
            "<quuz>",
            "foo",
            "bar",
            "baz",
            "qux",
            "quux",
            "<corge>",
        ]

    def test_pad_left(self):
        assert list(pad(tokenize("foo bar baz qux quux"), pad_left=False)) == [
            "foo",
            "bar",
            "baz",
            "qux",
            "quux",
            "</s>",
        ]

    def test_pad_right(self):
        assert list(pad(tokenize("foo bar baz qux quux"), pad_right=False)) == [
            "<s>",
            "foo",
            "bar",
            "baz",
            "qux",
            "quux",
        ]


class TestNgrams:
    def test_unigrams(self):
        assert list(ngrams(1, tokenize("foo bar baz qux quux"))) == [
            ("foo",),
            ("bar",),
            ("baz",),
            ("qux",),
            ("quux",),
        ]

    def test_bigrams(self):
        assert list(ngrams(2, tokenize("foo bar baz qux quux"))) == [
            ("foo", "bar"),
            ("bar", "baz"),
            ("baz", "qux"),
            ("qux", "quux"),
        ]

    def test_trigrams(self):
        assert (list(ngrams(3, tokenize("foo bar baz qux quux")))) == [
            ("foo", "bar", "baz"),
            ("bar", "baz", "qux"),
            ("baz", "qux", "quux"),
        ]


def test_train_test_split():
    assert train_test_split(tokenize("foo bar baz qux quux")) == (
        ["foo", "baz", "quux", "bar"],
        ["qux"],
    )
