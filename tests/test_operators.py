"""Tests for HAEA operator strategy edge cases."""

import numpy as np
import pytest

from pyecsago import HAEA, AdaptiveGaussianMutation, LinearCrossoverPerDimension
from pyecsago.core.exceptions import ConfigurationError


@pytest.fixture
def haea(cpu_context):
    return HAEA(
        mutation_operators={'mut': AdaptiveGaussianMutation},
        crossover_operators={'xover': LinearCrossoverPerDimension},
        context=cpu_context,
    )


class TestHAEAEdgeCases:
    def test_get_operator_arity_invalid(self, haea):
        with pytest.raises(ConfigurationError, match="Invalid operator"):
            haea.get_operator_arity("nonexistent")

    def test_apply_operator_invalid(self, haea, sample_individual):
        with pytest.raises(ConfigurationError, match="Invalid operator"):
            haea.apply_operator(
                haea.context.dtype_strategy,
                "nonexistent",
                sample_individual,
            )

    def test_apply_crossover_without_parent2(self, haea, sample_individual):
        with pytest.raises(ConfigurationError, match="Invalid operator"):
            haea.apply_operator(
                haea.context.dtype_strategy,
                "xover",
                sample_individual,
                parent2=None,
            )

    def test_adjust_rates_penalize(self, haea):
        np.random.seed(42)
        rates = {'mut': 0.5, 'xover': 0.5}
        result = haea.adjust_rates(rates, 'mut', reward=False)
        assert result['mut'] < 0.5

    def test_adjust_rates_reward(self, haea):
        np.random.seed(42)
        rates = {'mut': 0.5, 'xover': 0.5}
        result = haea.adjust_rates(rates, 'mut', reward=True)
        assert result['mut'] > 0.5

    def test_select_operator(self, haea):
        rates = {'mut': 0.9, 'xover': 0.1}
        op = haea.select_operator(rates)
        assert op in ('mut', 'xover')

    def test_apply_mutation(self, haea, sample_individual):
        offspring = haea.apply_operator(
            haea.context.dtype_strategy,
            'mut',
            sample_individual,
        )
        assert len(offspring) == 1

    def test_apply_crossover(self, haea, sample_individual):
        parent2 = sample_individual  # same individual is fine for testing
        offspring = haea.apply_operator(
            haea.context.dtype_strategy,
            'xover',
            sample_individual,
            parent2=parent2,
        )
        assert len(offspring) >= 1
