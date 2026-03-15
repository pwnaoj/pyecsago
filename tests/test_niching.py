"""Tests for niching strategies."""

import numpy as np
import pytest
from unittest.mock import patch

from pyecsago import HAEA, DeterministicCrowding, AdaptiveGaussianMutation, LinearCrossoverPerDimension


@pytest.fixture
def dc(cpu_context):
    haea = HAEA(
        mutation_operators={'mut': AdaptiveGaussianMutation},
        crossover_operators={'xover': LinearCrossoverPerDimension},
        context=cpu_context,
    )
    return DeterministicCrowding(operators_strategy=haea, context=cpu_context)


class TestDeterministicCrowding:
    def test_select_parents_unary(self, dc, sample_population):
        parents = dc.select_parents(sample_population[0], sample_population, 'mut')
        assert len(parents) == 1
        assert parents[0] is sample_population[0]

    def test_select_parents_binary(self, dc, sample_population):
        parents = dc.select_parents(sample_population[0], sample_population, 'xover')
        assert len(parents) == 2
        assert parents[0] is sample_population[0]

    def test_replace_default(self, dc):
        offspring = [1, 2, 3]
        result = dc.replace([], offspring)
        assert result == offspring

    def test_select_parents_none_arity(self, dc, sample_population):
        with patch.object(dc.operators_strategy, 'get_operator_arity', return_value=None):
            with pytest.raises(ValueError, match="no defined arity"):
                dc.select_parents(sample_population[0], sample_population, 'mut')
