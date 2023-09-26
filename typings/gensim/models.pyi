# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=too-few-public-methods
# pylint: disable=unused-argument

from typing import TYPE_CHECKING
from numpy import float64
from numpy.typing import NDArray

if TYPE_CHECKING:
    class KeyedVectors:
        index_to_key: list[str]
        key_to_index: dict[str, int]
        vectors: NDArray[float64]

        def __contains__(self, key: str) -> bool: ...
        def __getitem__(
            self, key_or_keys: str | list[str] | int | list[int]
        ) -> NDArray[float64]: ...
        def similar_by_vector(
            self, vector: NDArray[float64], topn: int = ...
        ) -> list[tuple[str, float]]: ...
