"""Base class for evolutionary algorithms."""

from abc import ABC, abstractmethod


class EvolutionaryAlgorithm(ABC):
    """Abstract base class defining the evolutionary algorithm interface.

    Subclasses must implement evolve, extract_prototypes, and refine_prototypes.
    """

    def __init__(self, config: dict):
        """Initializes the algorithm with configuration.

        Args:
            config: Algorithm configuration dictionary.
        """
        self.config: dict = config
        self.use_cuda: bool = config.get('use_cuda', False)

    @abstractmethod
    def evolve(self, max_generations: int = 100):
        """Runs the main evolutionary loop.

        Args:
            max_generations: Maximum number of generations.
        """
        pass

    @abstractmethod
    def extract_prototypes(self):
        """Extracts final prototypes from the evolved population."""
        pass

    @abstractmethod
    def refine_prototypes(self, prototypes, iterations: int = 10):
        """Refines extracted prototypes.

        Args:
            prototypes: Prototypes to refine.
            iterations: Number of refinement iterations.

        Returns:
            List of refined prototypes.
        """
        pass
