"""Utilities to load HuggingFace datasets."""

# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownVariableType=false

from collections.abc import Generator
from typing import TypedDict
from datasets import Dataset, load_dataset

CACHE_DIR = "./data_cache"


class Document(TypedDict):
    """A document item in a dataset."""

    text: str
    label: int


class HuggingFaceDataset:
    """A wrapper around HuggingFace datasets."""

    def __init__(
        self, path: str, name: str, split: str, cache_dir: str = CACHE_DIR
    ):
        self._path = path
        self._name = name
        self._split = split
        self._cache_dir = cache_dir

    @property
    def load(self) -> Dataset:
        """Load the dataset."""

        dataset = load_dataset(
            path=self._path,
            name=self._name,
            split=self._split,
            verification_mode="no_checks",
            cache_dir=self._cache_dir,
        )
        assert isinstance(dataset, Dataset)
        return dataset


class TweetEvalDataset(HuggingFaceDataset):
    """A wrapper around the "tweet_eval" dataset."""

    def __init__(self, name: str, split: str, cache_dir: str = CACHE_DIR):
        super().__init__(
            path="tweet_eval",
            name=name,
            split=split,
            cache_dir=cache_dir,
        )

    def iter(self) -> Generator[Document, None, None]:
        """Iterate the items in the dataset."""

        for item in self.load:
            assert isinstance(item, dict)

            text = item["text"]
            assert isinstance(text, str)

            label = item["label"]
            assert isinstance(label, int)

            yield Document(text=text, label=label)
