"""HAEA (Hybrid Adaptive Evolutionary Algorithm) operator strategy."""

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

from .genetic_operators import MutationOperator, CrossoverOperator
from .base import EvolutionaryOperator
from ...core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from pyecsago.core.context import AlgorithmContext
    from pyecsago.utils.data_types import DataTypeStrategy


class HAEA(EvolutionaryOperator):
    """Manages operator selection and adaptive rate adjustment.

    Delegates actual operator application to specialized mutation
    and crossover classes.
    """

    def __init__(self,
                 mutation_operators: dict[str, type[MutationOperator]],
                 crossover_operators: dict[str, type[CrossoverOperator]],
                 context: AlgorithmContext):
        """Initializes HAEA with available operators.

        Args:
            mutation_operators: Mapping of name to mutation operator class.
            crossover_operators: Mapping of name to crossover operator class.
            context: Algorithm execution context.
        """
        self.context = context

        self.mutation_operators = {
            name: operator() for name, operator in mutation_operators.items()
        }
        self.crossover_operators = {
            name: operator() for name, operator in crossover_operators.items()
        }

    def get_operator_arity(self, operator: str) -> int | None:
        """Returns the arity of the specified operator.

        Args:
            operator: Operator name.

        Returns:
            Arity (1 for mutation, 2 for crossover).

        Raises:
            ConfigurationError: If operator name is not found.
        """
        if operator in self.mutation_operators:
            return self.mutation_operators[operator].arity
        elif operator in self.crossover_operators:
            return self.crossover_operators[operator].arity
        else:
            raise ConfigurationError(f"Invalid operator: {operator}")

    def select_operator(self, operator_rates: dict) -> str:
        """Selects an operator probabilistically based on rates.

        Args:
            operator_rates: Mapping of operator name to selection probability.

        Returns:
            Name of the selected operator.
        """
        operators = list(operator_rates.keys())
        probabilities = list(operator_rates.values())
        return self.context.dtype_strategy.random_choice(operators, p=probabilities)

    def apply_operator(self, dtype_strategy: DataTypeStrategy, operator: str, parent1: object, parent2: object | None = None) -> list:
        """Applies the selected operator to produce offspring.

        Args:
            dtype_strategy: Data type strategy for array operations.
            operator: Name of the operator to apply.
            parent1: First parent individual.
            parent2: Second parent (required for crossover).

        Returns:
            List of offspring individuals.

        Raises:
            ConfigurationError: If operator is invalid or parents are insufficient.
        """
        if operator in self.mutation_operators:
            return [self.mutation_operators[operator].mutate(parent1, self.context.dtype_strategy)]

        elif operator in self.crossover_operators and parent2 is not None:
            return self.crossover_operators[operator].crossover(parent1, parent2, self.context.dtype_strategy)

        else:
            raise ConfigurationError(f"Invalid operator or insufficient parents: {operator}")

    def adjust_rates(self, operator_rates: dict[str, float], operator: str, reward: bool) -> dict[str, float]:
        """Adjusts operator rates based on performance.

        Args:
            operator_rates: Current operator rates to modify.
            operator: Name of the selected operator.
            reward: True to reward, False to penalize.

        Returns:
            Updated operator rates dictionary.
        """
        delta = self.context.dtype_strategy.module.random.random()

        if reward:
            operator_rates[operator] *= (1.0 + delta)
        else:
            operator_rates[operator] *= (1.0 - delta)

        return operator_rates
