"""Crossover operator implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .genetic_operators import CrossoverOperator


if TYPE_CHECKING:
    from pyecsago import ECSAGOIndividual
    from pyecsago.utils.data_types import DataTypeStrategy


class LinearCrossoverPerDimension(CrossoverOperator):
    """Linear crossover operator with independent alpha per dimension."""

    def crossover(self, parent1: ECSAGOIndividual, parent2: ECSAGOIndividual, dtype_strategy: DataTypeStrategy) -> list[ECSAGOIndividual]:
        """Performs linear crossover with an independent alpha for each dimension.

        Uses a different blending factor for each feature, allowing a more
        flexible combination of parent characteristics.
        """
        alphas = dtype_strategy.module.random.rand(len(parent1.genome))

        new_genome1 = dtype_strategy.module.multiply(alphas, parent1.genome) + \
                      dtype_strategy.module.multiply(1 - alphas, parent2.genome)
        new_genome2 = dtype_strategy.module.multiply(1 - alphas, parent1.genome) + \
                      dtype_strategy.module.multiply(alphas, parent2.genome)

        child1 = type(parent1)(
            genome=new_genome1,
            fitness_calculator=parent1.fitness_calculator,
            operator_rates=parent1.operator_rates.copy(),
            context=parent1.context
        )

        child2 = type(parent1)(
            genome=new_genome2,
            fitness_calculator=parent2.fitness_calculator,
            operator_rates=parent2.operator_rates.copy(),
            context=parent2.context
        )

        return [child1, child2]
