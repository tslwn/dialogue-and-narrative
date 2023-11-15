"""N-gram language model with optional add-k smoothing."""

from collections import Counter
from collections.abc import Callable
from math import log2
from typing import TypeVar
from numpy.random import multinomial
from .preprocessing import flatten, ngrams, pad

T = TypeVar("T")


class NgramModel:
    """A class for N-gram language models with optional add-k smoothing."""

    def __init__(
        self,
        order: int,
        k: float = 0.0,
        pad_fn: Callable[[list[str]], list[str]] | None = None,
    ) -> None:
        assert order >= 1, "Order must be at least 1."
        assert k >= 0.0, "k must be at least 0."

        self.order = order
        self.k = k
        self.pad = pad_fn or pad

        self.vocab: Counter[str] = Counter()
        self.context_counts: dict[tuple[str, ...], Counter[str]] = {}
        self.counts: Counter[tuple[str, ...]] = Counter()

    def fit(self, docs: list[list[str]]) -> None:
        """Fit the model to a list of documents."""
        for doc in docs:
            for token in self.pad(doc):
                self.vocab[token] += 1

        if self.order > 1:
            for ngram in ngrams(
                self.order, flatten(self.pad(doc) for doc in docs)
            ):
                self.counts[ngram] += 1
                context = maybe_tuple(ngram[:-1])
                self.context_counts.setdefault(context, Counter())[
                    ngram[-1]
                ] += 1

    def score(
        self, token: str, context: tuple[str, ...] | None = None
    ) -> float:
        """Return the probability of a token given an optional context."""
        num: int
        den: int

        # If there is no context, return the unigram probability.
        if context is None or self.order == 1:
            num = self.vocab[token]
            den = sum(self.vocab.values())
        # If there is context, return the N-gram probability.
        else:
            num = self.counts[(context + (token,))]
            den = sum(self.context_counts.get(context, Counter()).values())

        return (num + self.k) / (den + self.k * len(self.vocab))

    def logscore(
        self, token: str, context: tuple[str, ...] | None = None
    ) -> float:
        """Return the log probability of a token given an optional context."""
        return log2(self.score(token=token, context=context))

    def entropy(self, sequence: list[str]) -> float:
        """Return the entropy of a sequence."""
        return -sum(
            self.logscore(ngram[-1], maybe_tuple(ngram[-self.order + 1 :]))
            for ngram in ngrams(self.order, sequence)
        ) / len(list(sequence))

    def perplexity(self, sequence: list[str]) -> float:
        """Return the perplexity of a sequence."""
        return 2 ** self.entropy(sequence)

    def generate(
        self, num_words: int = 1, seed: list[str] | None = None
    ) -> list[str]:
        """
        Generate a sequence of tokens. Adapted from:
        https://www.nltk.org/_modules/nltk/lm/api.html#LanguageModel.generate.
        """
        seed = seed or []
        vocab = self.vocab.keys()

        if num_words == 1:
            if len(seed) >= self.order:
                context = tuple(seed[-self.order + 1 :])
                scores = [
                    (token, self.score(token, context)) for token in vocab
                ]
            else:
                scores = [(token, self.score(token)) for token in vocab]

            scores = sorted(scores, key=lambda sample: sample[1], reverse=True)
            return [scores[multinomial(1, [p for _, p in scores]).argmax()][0]]

        generated: list[str] = []
        for _ in range(num_words):
            generated = generated + self.generate(
                num_words=1,
                seed=seed + generated,
            )
        return generated


def maybe_tuple(value: tuple[T, ...] | T) -> tuple[T, ...]:
    """Return a tuple if the value is not already a tuple."""
    return value if isinstance(value, tuple) else (value,)  # type: ignore
