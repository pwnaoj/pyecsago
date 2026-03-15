"""Data type strategies for NumPy and CuPy interoperability."""

from __future__ import annotations

import types

import numpy as np

from abc import ABC, abstractmethod
from typing import Any


class DataTypeStrategy(ABC):
    """Abstract strategy for data type handling in pyecsago.

    Provides a consistent interface for array and scalar operations
    across CPU (NumPy) and GPU (CuPy) implementations.
    """

    @property
    @abstractmethod
    def module(self) -> types.ModuleType:
        """Returns the underlying module (numpy or cupy)."""
        pass

    @abstractmethod
    def array(self, data: Any) -> Any:
        """Converts data to the appropriate array type (np.ndarray or cp.ndarray).

        Args:
            data: Data to convert to an array.

        Returns:
            Array of the appropriate type.
        """
        pass

    @abstractmethod
    def scalar(self, value: float) -> Any:
        """Converts a scalar value to the appropriate type.

        Args:
            value: Scalar value to convert.

        Returns:
            Scalar of the appropriate type.
        """
        pass

    @abstractmethod
    def is_array_type(self, data: Any) -> bool:
        """Checks whether the data is of the array type managed by this strategy.

        Args:
            data: Data to check.

        Returns:
            True if the data matches this strategy's array type.
        """
        pass

    @abstractmethod
    def to_numpy(self, data: Any) -> np.ndarray:
        """Converts data to a numpy array (if necessary).

        Args:
            data: Data to convert.

        Returns:
            Data as a numpy array.
        """
        pass

    @abstractmethod
    def random_choice(self, a: Any, size: int | None = None, replace: bool = True, p: Any = None) -> Any:
        """Compatible random.choice implementation for both strategies."""
        pass

class NumPyStrategy(DataTypeStrategy):
    """Data type strategy using NumPy."""

    @property
    def module(self) -> types.ModuleType:
        """Returns the NumPy module."""
        return np

    def array(self, data: Any) -> np.ndarray:
        """Converts data to numpy.ndarray with float64 dtype."""
        return np.asarray(data, dtype=np.float64)

    def scalar(self, value: float) -> np.float64:
        """Converts a scalar value to numpy.float64."""
        return np.float64(value)

    def is_array_type(self, data: Any) -> bool:
        """Checks whether data is a numpy.ndarray."""
        return isinstance(data, np.ndarray)

    def to_numpy(self, data: Any) -> np.ndarray:
        """Converts data to numpy.ndarray (returns as-is if already one)."""
        return self.array(data) if not self.is_array_type(data) else data

    def random_choice(self, a: Any, size: int | None = None, replace: bool = True, p: Any = None) -> Any:
        """Uses numpy.random.choice directly."""
        return np.random.choice(a, size=size, replace=replace, p=p)

class CuPyStrategy(DataTypeStrategy):  # pragma: no cover
    """Data type strategy using CuPy."""

    def __init__(self):
        """Initializes the CuPy strategy via lazy import."""
        from .compat import get_cupy
        self.cp = get_cupy()

    @property
    def module(self):
        """Returns the CuPy module."""
        return self.cp

    def array(self, data):
        """Converts data to cupy.ndarray with float64 dtype."""
        if self.is_array_type(data):
            return data
        elif isinstance(data, np.ndarray):
            return self.cp.asarray(data, dtype=self.cp.float64)
        elif isinstance(data, list):
            data_ = [value.get() if hasattr(value, 'get') else value for value in data]
            return self.cp.array(data_, dtype=self.cp.float64)
        else:
            return self.cp.array(data, dtype=self.cp.float64)

    def scalar(self, value):
        """Converts a scalar value to cupy.float64."""
        return self.cp.float64(value)

    def is_array_type(self, data):
        """Checks whether data is a cupy.ndarray."""
        return hasattr(data, 'device') or isinstance(data, self.cp.ndarray)

    def to_numpy(self, data):
        """Converts data to numpy.ndarray (downloads from GPU if necessary)."""
        if self.is_array_type(data):
            return data.get()
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data, dtype=np.float64)

    def random_choice(self, a, size=None, replace=True, p=None):
        """Compatible random.choice that handles types unsupported by CuPy.

        CuPy does not support arrays of Python objects, so for object lists
        (individuals, strings, etc.) selection is done by index using NumPy.
        """
        if isinstance(a, (list, tuple)):
            # Convert p to NumPy if on GPU
            if p is not None:
                np_p = [prob.get() if self.is_array_type(prob) else prob for prob in p]
            else:
                np_p = None

            # Select indices with NumPy and return the corresponding objects
            indices = np.random.choice(len(a), size=size, replace=replace, p=np_p)
            if size is None:
                return a[indices]
            return [a[i] for i in indices]

        # If 'a' is a numeric array supported by CuPy
        return self.cp.random.choice(a, size=size, replace=replace, p=p)

class DataTypeStrategyFactory:
    """Factory for creating data type strategies."""

    @staticmethod
    def create_strategy(use_cuda: bool) -> DataTypeStrategy:
        """Creates the appropriate data type strategy.

        Args:
            use_cuda: Whether to use CuPy (True) or NumPy (False).

        Returns:
            Appropriate DataTypeStrategy instance.
        """
        if use_cuda:
            return CuPyStrategy()
        else:
            return NumPyStrategy()
