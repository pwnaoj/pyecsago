"""Base class for populations."""

from abc import ABC, abstractmethod


class BasePopulation(ABC):
    """Abstract base class for a population of individuals."""

    @abstractmethod
    def evaluate_population(self):
        """Evaluates fitness for all individuals in the population."""
        pass

    @abstractmethod
    def evolve(self):
        """Evolves the population to the next generation."""
        pass
