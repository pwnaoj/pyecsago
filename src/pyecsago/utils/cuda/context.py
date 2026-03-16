"""Persistent CUDA context for ECSAGO GPU operations."""

from __future__ import annotations

import cupy as cp

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


class CUDAContext:
    """Persistent CUDA context that keeps data on GPU across algorithm iterations."""

    def __init__(self, data: Any = None) -> None:
        """Initializes the CUDA context with optional data.

        Args:
            data: Dataset to load onto GPU (optional).
        """
        self.device = None
        self.stream = None
        self.data_gpu = None

        if data is not None:
            self.load_data(data)

    def load_data(self, data: Any) -> None:
        """Transfers a dataset to GPU memory.

        Args:
            data: Dataset to transfer.
        """
        if self.device is None:
            self.device = cp.cuda.Device()

        if isinstance(data, cp.ndarray):
            self.data_gpu = data
        else:
            self.data_gpu = cp.asarray(data, dtype=cp.float64)

    @contextmanager
    def get_context(self) -> Generator[CUDAContext, None, None]:
        """Provides a CUDA context with device and stream initialization."""
        try:
            if self.device is None:
                self.device = cp.cuda.Device()

            if self.stream is None:
                self.stream = cp.cuda.Stream(non_blocking=True)

            yield self
        finally:
            if self.stream is not None:
                self.stream.synchronize()

    def __del__(self) -> None:
        """Releases GPU resources."""
        try:
            if hasattr(self, 'data_gpu') and self.data_gpu is not None:
                del self.data_gpu

            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
