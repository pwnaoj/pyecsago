"""Base evolution strategy interface."""

from abc import ABC, abstractmethod


class EvolutionStrategy(ABC):
    """Base interface for evolution strategies."""

    @abstractmethod
    def evolve_population(self, population: list, data: list) -> list:
        """Evolves the current population using the defined strategy.

        Args:
            population: List of individuals forming the current population.
            data: Data needed for evaluation.

        Returns:
            New evolved population.
        """
        pass
