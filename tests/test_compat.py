"""Tests for pyecsago.utils.compat."""

import pytest

import pyecsago.utils.compat as compat


class TestCompat:
    def setup_method(self):
        # Reset cached state before each test
        compat._cupy = None
        compat._cupy_available = None

    def test_is_cupy_available_returns_bool(self):
        result = compat.is_cupy_available()
        assert isinstance(result, bool)

    def test_is_cupy_available_caches(self):
        result1 = compat.is_cupy_available()
        result2 = compat.is_cupy_available()
        assert result1 == result2

    def test_is_cupy_not_available(self):
        # In a test environment without CUDA, cupy should not be available
        assert compat.is_cupy_available() is False

    def test_get_cupy_raises_without_cuda(self):
        with pytest.raises(ImportError, match="CuPy is required"):
            compat.get_cupy()
