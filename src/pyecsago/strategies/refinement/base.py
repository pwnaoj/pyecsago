"""Base refinement strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import cupy as cp
    from pyecsago.implementations.ecsago.individual import ECSAGOIndividual
    from pyecsago.utils.data_types import DataTypeStrategy


class Refinement(ABC):
    """Base strategy for prototype refinement."""

    @abstractmethod
    def iteration(self, prototypes: list[ECSAGOIndividual], data: np.ndarray | cp.ndarray, dtype_strategy: DataTypeStrategy, **kwargs: Any) -> list[ECSAGOIndividual]:
        """Executes a single refinement iteration.

        Args:
            prototypes: List of prototypes to refine.
            data: Original dataset.
            dtype_strategy: Data type strategy (NumPy/CuPy).
            **kwargs: Additional refinement-specific parameters.

        Returns:
            List of refined prototypes after one iteration.
        """
        pass

    @abstractmethod
    def apply(self, prototypes: list[ECSAGOIndividual], data: np.ndarray | cp.ndarray, iterations: int, dtype_strategy: DataTypeStrategy, **kwargs: Any) -> list[ECSAGOIndividual]:
        """Applies the full refinement process.

        Args:
            prototypes: List of prototypes to refine.
            data: Original dataset.
            iterations: Number of refinement iterations.
            dtype_strategy: Data type strategy.
            **kwargs: Additional refinement-specific parameters.

        Returns:
            List of refined prototypes.
        """
        pass

    def init(self) -> None:
        """Initializes the refinement algorithm.

        May be overridden by concrete implementations for specific initialization.
        """
        pass

    def get_result(self) -> object | None:
        """Returns the refinement results.

        Returns:
            Refinement results or None.
        """
        return None
