"""pyecsago/utils/cuda/context.py"""

from __future__ import annotations

import cupy as cp

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


class CUDAContext:
    """
    Contexto CUDA con persistencia de datos para ECSAGO.

    Mantiene el conjunto de datos en la memoria GPU durante todo el ciclo de vida del algoritmo,
    evitando transferencias repetidas de CPU a GPU.
    """

    def __init__(self, data: Any = None) -> None:
        """
        Inicializa el contexto CUDA con datos opcionales.

        Args:
            data: Conjunto de datos a cargar en la GPU (opcional)
        """
        self.device = None
        self.stream = None
        self.data_gpu = None

        # Si se proporcionan datos, cargarlos inmediatamente
        if data is not None:
            self.load_data(data)

    def load_data(self, data: Any) -> None:
        """
        Carga un conjunto de datos en la memoria GPU.

        Args:
            data: Conjunto de datos a transferir a GPU
        """
        # Asegurar que el dispositivo está inicializado
        if self.device is None:
            self.device = cp.cuda.Device()

        # Cargar datos en GPU
        if isinstance(data, cp.ndarray):
            self.data_gpu = data
        else:
            self.data_gpu = cp.asarray(data, dtype=cp.float64)

    @contextmanager
    def get_context(self) -> Generator[CUDAContext, None, None]:
        """
        Proporciona un contexto CUDA para operaciones GPU.

        Este contexto garantiza que el dispositivo y stream estén
        correctamente inicializados y que el stream se sincronice al salir.
        """
        try:
            # Asegurar que el dispositivo está inicializado
            if self.device is None:
                self.device = cp.cuda.Device()

            # Crear stream si no existe
            if self.stream is None:
                self.stream = cp.cuda.Stream(non_blocking=True)

            yield self
        finally:
            # Sincronizar el stream antes de salir
            if self.stream is not None:
                self.stream.synchronize()

    def __del__(self) -> None:
        """Libera los recursos GPU cuando se destruye el contexto."""
        try:
            if hasattr(self, 'data_gpu') and self.data_gpu is not None:
                del self.data_gpu

            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
