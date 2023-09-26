"""Mathematical utilities."""

from typing import Any
from numpy import dot, ndarray
from numpy.linalg import norm


def similarity(vector1: ndarray[Any, Any], vector2: ndarray[Any, Any]) -> float:
    """The cosine similarity between two vectors."""
    return dot(
        vector1,
        vector2,
    ) / (norm(vector1) * norm(vector2))
