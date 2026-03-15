"""validators.py"""
from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike


class DataValidator:
    """Validates input data for the ECSAGO algorithm.

    Centralizes all data-related validations before usage
    by the main algorithm classes.
    """
    @staticmethod
    def validate_dataset(datos: np.ndarray | ArrayLike | None,
                        dimensiones: int | None = None) -> np.ndarray:
        """
        Valida que el conjunto de datos cumpla con los requisitos necesarios.
        
        Args:
            datos: Conjunto de datos a validar
            dimensiones: Dimensionalidad esperada de los datos (opcional)
            
        Returns:
            np.ndarray: Datos validados y convertidos a numpy array
            
        Raises:
            ValueError: Si los datos son None o no cumplen con los requisitos
            TypeError: Si los datos no son del tipo esperado
        """
        if datos is None:
            raise ValueError("El conjunto de datos no puede ser None")
            
        try:
            datos_array = np.asarray(datos)
        except (TypeError, ValueError):  # pragma: no cover
            raise TypeError("Los datos deben ser convertibles a numpy array")
            
        if datos_array.size == 0:
            raise ValueError("El conjunto de datos no puede estar vacío")
            
        if dimensiones is not None and datos_array.shape[1] != dimensiones:
            raise ValueError(f"La dimensionalidad de los datos ({datos_array.shape[1]}) " 
                           f"no coincide con la esperada ({dimensiones})")
            
        return datos_array
    