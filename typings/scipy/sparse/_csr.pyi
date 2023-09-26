# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=too-few-public-methods
# pylint: disable=unused-argument

from typing import TYPE_CHECKING, Self
from numpy import float64
from numpy.typing import NDArray

if TYPE_CHECKING:
    class csr_matrix:
        def getcol(self, index: int) -> Self: ...
        def getrow(self, index: int) -> Self: ...
        def toarray(self) -> NDArray[float64]: ...
