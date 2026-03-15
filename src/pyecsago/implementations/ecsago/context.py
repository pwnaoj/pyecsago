"""Standard algorithm context for CPU/GPU execution."""

from __future__ import annotations

import numpy as np

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from ...core.context import AlgorithmContext
from ...core.exceptions import ValidationError
from ...utils.data_types import DataTypeStrategyFactory

if TYPE_CHECKING:
    import cupy as cp


class StandardAlgorithmContext(AlgorithmContext):
    """Unified execution context managing CPU and GPU resources.

    Handles data storage, sigma2 limit computation, and transparent
    CPU/GPU switching via the DataTypeStrategy abstraction.
    """

    def __init__(self, use_cuda: bool = False) -> None:
        """Initializes the algorithm context.

        Args:
            use_cuda: Whether to use CUDA acceleration.
        """
        self._use_cuda = use_cuda
        self._cuda_context = None
        self._data = None
        self._dtype_strategy = DataTypeStrategyFactory.create_strategy(use_cuda)

        self._sigma2_max = None
        self._sigma2_min = None
        self._sigma2_initial = None

    @property
    def dtype_strategy(self) -> Any:
        """Returns the active data type strategy (NumPy/CuPy)."""
        return self._dtype_strategy

    @property
    def use_cuda(self) -> bool:
        """Returns whether CUDA is enabled."""
        return self._use_cuda

    @property
    def cuda_context(self) -> Any | None:
        """Returns the CUDA context, or None if not available."""
        return self._cuda_context

    def _initialize_sigma_limits(self) -> None:
        """Computes sigma2 bounds from data range (Eq 2.16).

        Raises:
            ValidationError: If no data has been set.
        """
        if self._data is None:
            raise ValidationError("Cannot compute sigma limits without data")

        xp = self.dtype_strategy.module
        data_range = xp.max(self._data, axis=0) - xp.min(self._data, axis=0)
        sigma2_max = float(xp.sum(data_range ** 2)) / 40.0

        self._sigma2_max = self.dtype_strategy.array(sigma2_max)
        self._sigma2_min = self.dtype_strategy.array(sigma2_max / 100.0)
        self._sigma2_initial = self.dtype_strategy.array(sigma2_max / 10.0)

    def get_sigma2_max(self) -> Any:
        """Returns the maximum allowed sigma2 value.

        Raises:
            ValidationError: If sigma limits are not initialized.
        """
        if self._sigma2_max is None:
            raise ValidationError("Sigma limits not initialized. Set data first.")
        return self._sigma2_max

    def get_sigma2_min(self) -> Any:
        """Returns the minimum allowed sigma2 value.

        Raises:
            ValidationError: If sigma limits are not initialized.
        """
        if self._sigma2_min is None:
            raise ValidationError("Sigma limits not initialized. Set data first.")
        return self._sigma2_min

    def get_sigma2_initial(self) -> Any:
        """Returns the initial sigma2 value for new individuals.

        Raises:
            ValidationError: If sigma limits are not initialized.
        """
        if self._sigma2_initial is None:
            raise ValidationError("Sigma limits not initialized. Set data first.")
        return self._sigma2_initial

    def set_data(self, data: np.ndarray | cp.ndarray) -> None:
        """Loads data into the context and initializes sigma limits.

        Args:
            data: Input dataset for the algorithm.
        """
        self._release_resources()

        self._data = self.dtype_strategy.array(data)

        if self.use_cuda:  # pragma: no cover
            from ...utils.cuda.context import CUDAContext
            self._cuda_context = CUDAContext(self._data)

        self._initialize_sigma_limits()

    def get_data(self) -> np.ndarray | cp.ndarray:
        """Returns the current dataset.

        Raises:
            ValidationError: If no data has been set.
        """
        if self._data is None:
            raise ValidationError("No data set in context")

        if self.use_cuda and self.cuda_context is not None:  # pragma: no cover
            return self.cuda_context.data_gpu

        return self._data

    def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Executes a function in the appropriate context (CPU/GPU).

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.
        """
        if self.use_cuda and self.cuda_context is not None:  # pragma: no cover
            with self.cuda_context.get_context() as ctx:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def to_cpu(self, data: np.ndarray | cp.ndarray) -> np.ndarray:
        """Converts data to CPU (numpy) format.

        Args:
            data: Data to convert.

        Returns:
            Numpy array.
        """
        return self.dtype_strategy.to_numpy(data)

    def to_device(self, data: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        """Converts data to the current device format (CPU/GPU).

        Args:
            data: Data to convert.

        Returns:
            Array in the current device format.
        """
        return self.dtype_strategy.array(data)

    def _release_resources(self) -> None:
        """Releases allocated resources (data and CUDA context)."""
        self._data = None

        if self._cuda_context is not None:  # pragma: no cover
            try:
                del self._cuda_context
                self._cuda_context = None
            except Exception:
                pass

    def __del__(self) -> None:  # pragma: no cover
        """Ensures resource cleanup on destruction."""
        self._release_resources()
