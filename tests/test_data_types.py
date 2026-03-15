"""Tests for pyecsago.utils.data_types."""

import numpy as np
import pytest

from pyecsago.utils.data_types import NumPyStrategy, DataTypeStrategyFactory


class TestNumPyStrategy:
    def test_module(self):
        s = NumPyStrategy()
        assert s.module is np

    def test_array_from_list(self):
        s = NumPyStrategy()
        arr = s.array([1.0, 2.0, 3.0])
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64

    def test_array_from_ndarray(self):
        s = NumPyStrategy()
        original = np.array([1, 2, 3], dtype=np.int32)
        arr = s.array(original)
        assert arr.dtype == np.float64

    def test_scalar(self):
        s = NumPyStrategy()
        val = s.scalar(3.14)
        assert isinstance(val, np.float64)
        assert val == pytest.approx(3.14)

    def test_is_array_type_true(self):
        s = NumPyStrategy()
        assert s.is_array_type(np.array([1, 2])) is True

    def test_is_array_type_false(self):
        s = NumPyStrategy()
        assert s.is_array_type([1, 2]) is False

    def test_to_numpy_from_ndarray(self):
        s = NumPyStrategy()
        arr = np.array([1.0, 2.0])
        result = s.to_numpy(arr)
        assert result is arr  # should return same object

    def test_to_numpy_from_list(self):
        s = NumPyStrategy()
        result = s.to_numpy([1.0, 2.0])
        assert isinstance(result, np.ndarray)

    def test_random_choice(self):
        s = NumPyStrategy()
        np.random.seed(42)
        result = s.random_choice([10, 20, 30], size=2, replace=True)
        assert len(result) == 2

    def test_random_choice_with_probabilities(self):
        s = NumPyStrategy()
        np.random.seed(42)
        result = s.random_choice([10, 20, 30], size=5, replace=True, p=[0.0, 0.0, 1.0])
        assert all(r == 30 for r in result)


class TestDataTypeStrategyFactory:
    def test_create_numpy_strategy(self):
        s = DataTypeStrategyFactory.create_strategy(use_cuda=False)
        assert isinstance(s, NumPyStrategy)

    def test_create_cupy_strategy_raises_without_cuda(self):
        # CuPy is not available in test environment, so this should raise ImportError
        with pytest.raises(ImportError):
            DataTypeStrategyFactory.create_strategy(use_cuda=True)
