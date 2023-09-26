"""Utility functions for sequence tagging."""

import os
import pickle
from logging import Logger
from typing import Any, TypeVar
import numpy
import numpy.typing
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..log import create_logger

T = TypeVar("T")

Int = TypeVar("Int", bound=numpy.int_)

# pylint: disable=invalid-name
DType_co = TypeVar("DType_co", covariant=True, bound=numpy.generic)

Arr = numpy.ndarray[Any, numpy.dtype[DType_co]]


def flat_nested_list(nested_list: list[list[T]]) -> list[T]:
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in sublist]


def map_nested_list(
    nested_list: list[list[Any]],
    flat_list_map: Arr[Int],
) -> list[list[Int]]:
    """Map the values of a flat list to the a nested list by index."""
    nested_list_map: list[list[Int]] = []
    flat_list_index = 0
    for sublist in nested_list:
        sublist_map: list[Int] = []
        for _ in sublist:
            sublist_map.append(flat_list_map[flat_list_index])
            flat_list_index += 1
        nested_list_map.append(sublist_map)
    return nested_list_map


def pad_truncate_array(array: Arr[Int], length: int, value: Int):
    """Pad or truncate an array."""
    if len(array) > length:
        return array[:length]
    return numpy.pad(
        array,
        (0, length - len(array)),
        mode="constant",
        constant_values=(value,),
    )


def flat_pad_truncate_nested_list(
    nested_list: list[list[Int]], length: int, value: Int
) -> Arr[Int]:
    """Pad or truncate and flatten a nested list to an array."""
    subarrays = []
    for sublist in nested_list:
        subarrays.append(
            pad_truncate_array(numpy.array(sublist), length, value)
        )
    return numpy.stack(subarrays, axis=0)


def to_data_loader(
    input_array: Arr[Int],
    label_array: Arr[Int],
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Convert NumPy arrays to a PyTorch data-loader."""
    input_tensor = torch.from_numpy(input_array).long()
    label_tensor = torch.from_numpy(label_array).long()
    tensor_dataset = TensorDataset(input_tensor, label_tensor)
    loader = DataLoader[tuple[torch.Tensor, ...]](
        tensor_dataset, batch_size=batch_size, shuffle=shuffle
    )
    return loader


def pickler(file: str, pickle_logger: Logger = create_logger("pickle", "info")):
    """Cache the result of a function in a pickle file."""

    def decorator(fn):
        def wrapped(*args, **kwargs):
            if os.path.exists(file):
                with open(file, "rb") as handle:
                    pickle_logger.info("loading %s", file)
                    pickled = pickle.load(handle)
                    pickle_logger.info("loaded %s", file)
                    return pickled

            result = fn(*args, **kwargs)

            with open(file, "wb") as handle:
                pickle_logger.info("dumping %s", file)
                pickle.dump(result, handle)
                pickle_logger.info("dumped %s", file)

            return result

        return wrapped

    return decorator
