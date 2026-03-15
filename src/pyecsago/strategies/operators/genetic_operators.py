"""Base interfaces for genetic operators (mutation and crossover)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyecsago.core.individual import BaseIndividual


class MutationOperator(ABC):
    """Base interface for mutation operators.

    Mutation operators modify a single individual to explore the search space.
    """

    @abstractmethod
    def mutate(self, individual: BaseIndividual) -> BaseIndividual:
        """Applies mutation to an individual.

        Args:
            individual: The individual to mutate.

        Returns:
            New mutated individual.
        """
        pass

    @property
    def arity(self) -> int:
        """Mutation operators always have arity 1."""
        return 1

class CrossoverOperator(ABC):
    """Base interface for crossover operators.

    Crossover operators combine genetic information from two individuals
    to produce new offspring.
    """

    @abstractmethod
    def crossover(self, parent1: BaseIndividual, parent2: BaseIndividual) -> list[BaseIndividual]:
        """Performs crossover between two individuals.

        Args:
            parent1: First parent for crossover.
            parent2: Second parent for crossover.

        Returns:
            List of generated offspring.
        """
        pass

    @property
    def arity(self) -> int:
        """Crossover operators always have arity 2."""
        return 2
