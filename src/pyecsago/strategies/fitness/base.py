"""Base fitness calculator interface."""

import numpy as np

from abc import ABC, abstractmethod


class FitnessCalculator(ABC):
    """Base interface for fitness calculation."""

    @abstractmethod
    def calculate(self, individual, data: np.ndarray, weight_threshold: float,
                 metric: str = 'euclidean', p_minkowski: int = 2) -> float:
        """Calculates fitness for a given individual."""
        pass
