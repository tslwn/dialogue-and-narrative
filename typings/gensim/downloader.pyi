# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=unused-argument

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    def info(
        name: str | None = ...,
        show_only_latest: bool = ...,
        name_only: bool = ...,
    ) -> dict[Any, Any]: ...
    def load(name: str, return_path: bool = ...) -> Any: ...
