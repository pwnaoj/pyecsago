"""Base class for population individuals."""

from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import cupy as cp


class BaseIndividual(ABC):
    """Abstract base class for a population individual."""

    def __init__(self, genome: np.ndarray | cp.ndarray, sigma2: np.ndarray | cp.ndarray, **kwargs: Any) -> None:
        self.genome = genome
        self.sigma2 = sigma2
        self.fitness = 0.0

    @abstractmethod
    def calculate_fitness(self, data: np.ndarray, **kwargs: Any) -> None:
        """Calculates the individual's fitness."""
        pass

    def clone(self) -> BaseIndividual:
        """Creates a copy of the individual.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must override clone()")
