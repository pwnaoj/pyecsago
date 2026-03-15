"""Abstract context interface for evolutionary algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pyecsago.utils.data_types import DataTypeStrategy


class AlgorithmContext(ABC):
    """Abstract context managing data, type strategy, and CUDA resources."""

    @property
    @abstractmethod
    def dtype_strategy(self) -> DataTypeStrategy:
        """Returns the data type strategy (NumPy/CuPy)."""
        pass

    @property
    @abstractmethod
    def use_cuda(self) -> bool:
        """Returns whether CUDA is enabled."""
        pass

    @property
    @abstractmethod
    def cuda_context(self) -> Any | None:
        """Returns the CUDA context, or None if not available."""
        pass

    @abstractmethod
    def _initialize_sigma_limits(self) -> None:
        """Initializes sigma2 bounds based on current data."""
        pass

    @abstractmethod
    def get_sigma2_max(self) -> Any:
        """Returns the maximum allowed sigma2 value."""
        pass

    @abstractmethod
    def get_sigma2_min(self) -> Any:
        """Returns the minimum allowed sigma2 value."""
        pass

    @abstractmethod
    def get_sigma2_initial(self) -> Any:
        """Returns the initial sigma2 value for new individuals."""
        pass

    @abstractmethod
    def set_data(self, data: Any) -> None:
        """Sets the dataset in the context.

        Args:
            data: Dataset to set.
        """
        pass

    @abstractmethod
    def get_data(self) -> Any:
        """Returns the current dataset."""
        pass

    @abstractmethod
    def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Executes a function in the appropriate context (CPU/GPU).

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.
        """
        pass
