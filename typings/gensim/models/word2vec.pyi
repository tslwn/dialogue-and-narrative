# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=unused-argument

from typing import TYPE_CHECKING
from gensim.models import KeyedVectors

if TYPE_CHECKING:
    from _typeshed import Incomplete

    class Word2Vec:
        wv: KeyedVectors

        def __init__(
            self,
            sentences: list[list[str]] | None = ...,
            corpus_file: Incomplete | None = ...,
            vector_size: int = ...,
            alpha: float = ...,
            window: int = ...,
            min_count: int = ...,
            max_vocab_size: Incomplete | None = ...,
            sample: float = ...,
            seed: int = ...,
            workers: int = ...,
            min_alpha: float = ...,
            sg: int = ...,
            hs: int = ...,
            negative: int = ...,
            ns_exponent: float = ...,
            cbow_mean: int = ...,
            # hashfxn=...,
            epochs: int = ...,
            null_word: int = ...,
            trim_rule: Incomplete | None = ...,
            sorted_vocab: int = ...,
            # batch_words=...,
            compute_loss: bool = ...,
            # callbacks=...,
            comment: Incomplete | None = ...,
            max_final_vocab: Incomplete | None = ...,
            shrink_windows: bool = ...,
        ) -> None: ...
