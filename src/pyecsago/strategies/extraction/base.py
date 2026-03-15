"""Base extraction strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import cupy as cp
    from pyecsago.implementations.ecsago.individual import ECSAGOIndividual


class ExtractionStrategy(ABC):
    """Base strategy for prototype extraction."""

    @abstractmethod
    def extract(self, candidates: list[ECSAGOIndividual], data: np.ndarray | cp.ndarray) -> list[ECSAGOIndividual]:
        """Extracts prototypes from a list of candidates.

        Args:
            candidates: List of candidate individuals sorted by fitness (descending).
            data: Original dataset.

        Returns:
            List of extracted prototypes.
        """
        pass
