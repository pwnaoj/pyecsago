"""ECSAGO individual implementation."""

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

from ...core.individual import BaseIndividual
from ...core.exceptions import ValidationError

if TYPE_CHECKING:
    import cupy as cp
    from pyecsago.core.context import AlgorithmContext
    from pyecsago.strategies.fitness.base import FitnessCalculator


class ECSAGOIndividual(BaseIndividual):
    """Individual for the ECSAGO algorithm.

    Extends BaseIndividual with sigma2 dispersion management and
    adaptive operator rates for self-adaptive genetic operators.
    """

    def __init__(self,
                genome: np.ndarray | cp.ndarray,
                fitness_calculator: FitnessCalculator,
                operator_rates: dict[str, float],
                context: AlgorithmContext) -> None:
        """Initializes an ECSAGO individual.

        Args:
            genome: N-dimensional cluster center vector.
            fitness_calculator: Fitness calculator instance.
            operator_rates: Initial operator rates {name: rate}.
            context: Algorithm execution context.

        Raises:
            ValidationError: If context is None.
        """
        if context is None:
            raise ValidationError("A valid context is required to initialize ECSAGOIndividual.")

        self.context = context

        genome = self.context.dtype_strategy.array(genome)
        sigma2 = self.context.get_sigma2_initial()

        super().__init__(genome, sigma2)

        self.fitness_calculator = fitness_calculator
        self.operator_rates = operator_rates.copy()
        self._normalize_rates()
        self._validate_sigma2_limits()

    def get_radius(self, k: float = 13.8) -> float:
        """Computes the cluster radius as k * sigma2.

        Args:
            k: Chi-squared factor. Default 13.8 corresponds to
               chi2(2, 0.995) for 2D data.

        Returns:
            Cluster radius.
        """
        return k * self.sigma2

    def calculate_fitness(self, weight_threshold: float, metric: str = 'euclidean', p_minkowski: int = 2) -> float:
        """Computes and updates the individual's fitness.

        Args:
            weight_threshold: Threshold for weight binarization.
            metric: Distance metric ('euclidean', 'minkowski', 'cosine', 'jaccard').
            p_minkowski: Minkowski parameter (p=2 is euclidean, p=1 is manhattan).

        Returns:
            Computed fitness value.
        """
        data = self.context.get_data()

        fitness_value = self.fitness_calculator.calculate(
            self, data, weight_threshold, metric, p_minkowski
        )

        self.fitness = fitness_value

        return fitness_value

    def _validate_sigma2_limits(self) -> None:
        """Clips sigma2 to the allowed range [sigma2_min, sigma2_max].

        Raises:
            ValidationError: If context is None.
        """
        if self.context is None:
            raise ValidationError("A valid context is required to validate sigma2 limits.")

        sigma2_max = self.context.get_sigma2_max()
        sigma2_min = self.context.get_sigma2_min()
        sigma_value = self.context.dtype_strategy.array(self.sigma2)

        sigma_value_clipped = self.context.dtype_strategy.module.clip(sigma_value, sigma2_min, sigma2_max)

        self.sigma2 = sigma_value_clipped

    def _normalize_rates(self) -> None:
        """Normalizes operator rates to sum to 1.0."""
        rates_array = self.context.dtype_strategy.array(list(self.operator_rates.values()))

        normalized_rates = rates_array / self.context.dtype_strategy.module.sum(rates_array)

        self.operator_rates = dict(zip(self.operator_rates.keys(), normalized_rates))

    def clone(self) -> ECSAGOIndividual:
        """Creates a deep copy of this individual.

        Returns:
            New ECSAGOIndividual with copied attributes.
        """
        new_individual = type(self)(
            genome=self.genome.copy(),
            operator_rates=self.operator_rates.copy(),
            fitness_calculator=self.fitness_calculator,
            context=self.context
        )
        new_individual.sigma2 = self.sigma2
        new_individual.fitness = self.fitness

        return new_individual
