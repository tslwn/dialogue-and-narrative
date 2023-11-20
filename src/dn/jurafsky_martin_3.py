"""Solutions to the exercises in chapter 3."""

# pylint: disable=missing-function-docstring
# pyright: reportUnknownMemberType=false

from math import prod
from dn.bigram_language_model.ngram_model import NgramModel
from dn.bigram_language_model.preprocessing import ngrams, pad, tokenize


def exercise_3_4():
    corpus = [
        tokenize(doc)
        for doc in [
            "I am Sam",
            "Sam I am",
            "I am Sam",
            "I do not like green eggs and Sam",
        ]
    ]
    model = NgramModel(order=2, k=1.0)
    model.fit(corpus)
    print(f"Exercise 3.4: {model.score('Sam', ('am',)):.6f}")


def exercise_3_5():
    def pad_fn(
        sequence: list[str],
    ) -> list[str]:
        return list(pad(sequence, pad_right=False))

    def score(model: NgramModel, corpus: list[list[str]]) -> float:
        return sum(
            prod(
                model.score(ngram[-1], tuple(ngram[-model.order + 1 :]))
                for ngram in ngrams(model.order, pad_fn(sequence))
            )
            for sequence in corpus
        )

    corpus_1 = [
        tokenize(doc)
        for doc in [
            "a b",
            "b b",
            "b a",
            "a a",
        ]
    ]
    model = NgramModel(order=2, pad_fn=pad_fn)

    # Don't concatenate the documents because they don't have an end-symbol.
    model.fit([corpus_1[0]])
    model.fit([corpus_1[1]])
    model.fit([corpus_1[2]])
    model.fit([corpus_1[3]])

    print(f"Exercise 3.5.1: {score(model, corpus_1):.1f}")

    corpus_2 = [
        tokenize(doc)
        for doc in [
            "a b a",
            "b b a",
            "b a a",
            "a a a",
            "a b b",
            "b b b",
            "b a b",
            "a a b",
        ]
    ]

    print(f"Exercise 3.5.2: {score(model, corpus_2):.1f}")


def exercise_3_7():
    corpus = [
        tokenize(doc)
        for doc in [
            "I am Sam",
            "Sam I am",
            "I am Sam",
            "I do not like green eggs and Sam",
        ]
    ]
    model_1 = NgramModel(order=1)
    model_1.fit(corpus)
    model_2 = NgramModel(order=2)
    model_2.fit(corpus)
    print(
        f"Exercise 3.7: {0.5 * model_1.score('Sam') + 0.5 * model_2.score('Sam', ('am',)):.6f}"
    )


def exercise_3_10():
    corpus = [
        tokenize(doc)
        for doc in [
            "I am Sam",
            "Sam I am",
            "I am Sam",
            "I do not like green eggs and Sam",
        ]
    ]
    model = NgramModel(order=2)
    model.fit(corpus)
    seed: list[str] = []
    print(f"Exercise 3.10: {' '.join(seed + model.generate(10, seed))}")


if __name__ == "__main__":
    exercise_3_4()
    exercise_3_5()
    exercise_3_7()
    exercise_3_10()
