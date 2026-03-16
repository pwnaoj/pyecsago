"""validators.py"""
from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike


class DataValidator:
    """Validates input data for the ECSAGO algorithm."""

    @staticmethod
    def validate_dataset(data: np.ndarray | ArrayLike | None,
                        dimensions: int | None = None) -> np.ndarray:
        """Validates that the dataset meets the required constraints.

        Args:
            data: Dataset to validate.
            dimensions: Expected dimensionality (optional).

        Returns:
            Validated numpy array.

        Raises:
            ValueError: If data is None, empty, or has wrong dimensionality.
            TypeError: If data cannot be converted to a numpy array.
        """
        if data is None:
            raise ValueError("Dataset cannot be None")

        try:
            data_array = np.asarray(data)
        except (TypeError, ValueError):  # pragma: no cover
            raise TypeError("Data must be convertible to a numpy array")

        if data_array.size == 0:
            raise ValueError("Dataset cannot be empty")

        if dimensions is not None and data_array.shape[1] != dimensions:
            raise ValueError(f"Data dimensionality ({data_array.shape[1]}) "
                           f"does not match expected ({dimensions})")

        return data_array
