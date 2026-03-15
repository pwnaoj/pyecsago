"""Mutation operator implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .genetic_operators import MutationOperator


if TYPE_CHECKING:
    from pyecsago import ECSAGOIndividual
    from pyecsago.utils.data_types import DataTypeStrategy


class AdaptiveGaussianMutation(MutationOperator):
    """Adaptive Gaussian mutation operator for ECSAGO."""

    def mutate(self, individual: ECSAGOIndividual, dtype_strategy: DataTypeStrategy) -> ECSAGOIndividual:
        """Applies adaptive Gaussian mutation to a single feature.

        Randomly selects one feature and modifies it using a Gaussian
        distribution with adaptive variance based on sigma2.
        """
        sigma = dtype_strategy.module.sqrt(individual.sigma2)

        feature_to_mutate = dtype_strategy.module.random.randint(0, len(individual.genome))
        mutation_value = dtype_strategy.module.random.normal(0, sigma)

        mutation_vector = dtype_strategy.module.zeros_like(individual.genome)
        mutation_vector[feature_to_mutate] = mutation_value

        new_genome = individual.genome + mutation_vector
        new_individual = type(individual)(
            genome=new_genome,
            fitness_calculator=individual.fitness_calculator,
            operator_rates=individual.operator_rates.copy(),
            context=individual.context
        )

        return new_individual

class GaussianMutation(MutationOperator):
    """Standard Gaussian mutation operator for ECSAGO."""

    def mutate(self, individual: ECSAGOIndividual, dtype_strategy: DataTypeStrategy) -> ECSAGOIndividual:
        """Applies Gaussian mutation to a single feature.

        Args:
            individual: Individual to mutate.
        """
        sigma = dtype_strategy.module.sqrt(individual.sigma2)
        xsigma = sigma / 4.0

        component_to_mutate = dtype_strategy.module.random.randint(0, len(individual.genome))
        mutation_value = dtype_strategy.module.random.normal(0, xsigma)

        mutation_vector = dtype_strategy.module.zeros_like(individual.genome)
        mutation_vector[component_to_mutate] = mutation_value

        new_genome = individual.genome + mutation_vector
        new_individual = type(individual)(
            genome=new_genome,
            fitness_calculator=individual.fitness_calculator,
            operator_rates=individual.operator_rates.copy(),
            context=individual.context
        )

        return new_individual
