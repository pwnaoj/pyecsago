"""Tests for ECSAGOIndividual edge cases."""

import numpy as np
import pytest

from pyecsago import ECSAGOIndividual
from pyecsago.core.exceptions import ValidationError


class TestECSAGOIndividualEdgeCases:
    def test_none_context_raises(self, fitness_calculator, operator_rates):
        with pytest.raises(ValidationError, match="context"):
            ECSAGOIndividual(
                genome=np.array([0.0, 0.0]),
                fitness_calculator=fitness_calculator,
                operator_rates=operator_rates,
                context=None,
            )

    def test_get_radius(self, sample_individual):
        radius = sample_individual.get_radius(k=13.8)
        assert radius > 0

    def test_calculate_fitness(self, sample_individual):
        fitness = sample_individual.calculate_fitness(weight_threshold=0.3)
        assert float(fitness) >= 0.0

    def test_validate_sigma2_limits_none_context(self, sample_individual):
        sample_individual.context = None
        with pytest.raises(ValidationError, match="context"):
            sample_individual._validate_sigma2_limits()
