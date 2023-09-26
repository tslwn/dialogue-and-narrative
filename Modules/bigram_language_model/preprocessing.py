"""Preprocessing utilities."""
from collections.abc import Generator, Iterable
from typing import TypeVar
import random
import re

T = TypeVar("T")


def tokenize(string: str) -> list[str]:
    """
    A simple tokenizer that splits text with a regular expression.
    """
    return re.findall(r"\w+|\$[\d\.]+|\S+", string)


def flatten(
    sequence: Iterable[Iterable[T]] | Generator[Iterable[T], None, None],
) -> Iterable[T]:
    """
    Returns an iterator over the flattened sequence.
    """
    for item in sequence:
        yield from item


def pad(
    sequence: list[str],
    pad_left: bool = True,
    left_pad_symbol: str = "<s>",
    pad_right: bool = True,
    right_pad_symbol: str = "</s>",
) -> Iterable[str]:
    """
    Returns an iterator over the padded sequence.
    """
    if pad_left:
        yield left_pad_symbol
    for item in sequence:
        yield item
    if pad_right:
        yield right_pad_symbol


def ngrams(order: int, sequence: Iterable[T]) -> Iterable[tuple[T, ...]]:
    """
    Returns an iterator over N-grams.
    """
    sequence = list(sequence)
    for index in range(len(sequence) - order + 1):
        yield tuple(sequence[index : index + order])


def train_test_split(
    sequence: Iterable[T], train_size: float = 0.8, seed: int = 123
) -> tuple[list[T], list[T]]:
    """
    Returns a pseudo-random split of the sequence into train and test sequences.
    """
    sequence = list(sequence)

    # Return a new sequence in pseudo-random order.
    random.seed(seed)
    shuffled = random.sample(sequence, len(sequence))

    # Split the sequence into train and test sequences.
    index = int(len(shuffled) * train_size)
    return shuffled[:index], shuffled[index:]
