"""Tests for context and related uncovered lines."""

import numpy as np
import pytest

from pyecsago.implementations.ecsago.context import StandardAlgorithmContext
from pyecsago.core.exceptions import ValidationError


class TestContextExtended:
    def test_sigma_limits_without_data_raises(self):
        ctx = StandardAlgorithmContext(use_cuda=False)
        with pytest.raises(ValidationError, match="(?i)sigma"):
            ctx.get_sigma2_max()

    def test_sigma2_min_without_data_raises(self):
        ctx = StandardAlgorithmContext(use_cuda=False)
        with pytest.raises(ValidationError, match="(?i)sigma"):
            ctx.get_sigma2_min()

    def test_sigma2_initial_without_data_raises(self):
        ctx = StandardAlgorithmContext(use_cuda=False)
        with pytest.raises(ValidationError, match="(?i)sigma"):
            ctx.get_sigma2_initial()

    def test_get_data_without_data_raises(self):
        ctx = StandardAlgorithmContext(use_cuda=False)
        with pytest.raises(ValidationError, match="No data"):
            ctx.get_data()

    def test_to_cpu(self, cpu_context):
        data = cpu_context.get_data()
        result = cpu_context.to_cpu(data)
        assert isinstance(result, np.ndarray)

    def test_to_device(self, cpu_context):
        arr = [1.0, 2.0, 3.0]
        result = cpu_context.to_device(arr)
        assert isinstance(result, np.ndarray)

    def test_execute_with_args(self, cpu_context):
        result = cpu_context.execute(lambda x, y: x + y, 3, 4)
        assert result == 7

    def test_release_resources(self, cpu_context):
        cpu_context._release_resources()
        assert cpu_context._data is None

    def test_del(self):
        ctx = StandardAlgorithmContext(use_cuda=False)
        ctx.set_data(np.array([[1, 2], [3, 4]]))
        del ctx  # should not raise

    def test_initialize_sigma_limits_without_data(self):
        ctx = StandardAlgorithmContext(use_cuda=False)
        with pytest.raises(ValidationError, match="sigma limits without data"):
            ctx._initialize_sigma_limits()
