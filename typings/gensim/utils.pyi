# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=too-many-arguments
# pylint: disable=unused-argument

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    def tokenize(
        text: str,
        lowercase: bool = ...,
        deacc: bool = ...,
        encoding: str = ...,
        errors: str = ...,
        to_lower: bool = ...,
        lower: bool = ...,
    ) -> Sequence[str]: ...
