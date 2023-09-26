# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=redefined-builtin
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=unused-argument

from typing import TYPE_CHECKING
from numpy import str_
from numpy.typing import NDArray
from scipy.sparse._csr import csr_matrix

if TYPE_CHECKING:
    from _typeshed import Incomplete

    class CountVectorizer:
        def __init__(
            self,
            *,
            input: str = ...,
            encoding: str = ...,
            decode_error: str = ...,
            strip_accents: Incomplete | None = ...,
            lowercase: bool = ...,
            preprocessor: Incomplete | None = ...,
            tokenizer: Incomplete | None = ...,
            stop_words: Incomplete | None = ...,
            token_pattern: str = ...,
            ngram_range: tuple[int, int] = ...,
            analyzer: str = ...,
            max_df: float = ...,
            min_df: int = ...,
            max_features: Incomplete | None = ...,
            vocabulary: Incomplete | None = ...,
            binary: bool = ...,
            # dtype=...
        ) -> None: ...
        def fit(
            self, raw_documents: list[str], y: Incomplete | None = ...
        ) -> None: ...
        vocabulary_: dict[str, int]
        def fit_transform(
            self, raw_documents: list[str], y: Incomplete | None = ...
        ) -> csr_matrix: ...
        def transform(self, raw_documents: list[str]) -> csr_matrix: ...
        def get_feature_names_out(
            self, input_features: Incomplete | None = ...
        ) -> NDArray[str_]: ...
