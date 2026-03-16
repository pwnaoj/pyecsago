"""Tests for pyecsago.utils.validators."""

import numpy as np
import pytest

from pyecsago.utils.validators import DataValidator


class TestDataValidator:
    def test_validate_valid_array(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = DataValidator.validate_dataset(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_validate_list_input(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        result = DataValidator.validate_dataset(data)
        assert isinstance(result, np.ndarray)

    def test_validate_none_raises(self):
        with pytest.raises(ValueError, match="None"):
            DataValidator.validate_dataset(None)

    def test_validate_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            DataValidator.validate_dataset(np.array([]))

    def test_validate_correct_dimensions(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = DataValidator.validate_dataset(data, dimensions=2)
        assert result.shape[1] == 2

    def test_validate_wrong_dimensions_raises(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="dimensionality"):
            DataValidator.validate_dataset(data, dimensions=3)

    def test_validate_no_dimensions_check(self):
        data = np.array([[1.0, 2.0, 3.0]])
        result = DataValidator.validate_dataset(data)
        assert result.shape == (1, 3)

    def test_validate_convertible_input(self):
        # numpy can convert most things, so test with a valid tuple
        result = DataValidator.validate_dataset(((1.0, 2.0), (3.0, 4.0)))
        assert result.shape == (2, 2)
