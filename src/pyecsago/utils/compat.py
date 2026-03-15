"""Lazy import compatibility for optional CuPy dependency."""

from __future__ import annotations

import types

_cupy: types.ModuleType | None = None
_cupy_available: bool | None = None


def is_cupy_available() -> bool:
    """Checks whether CuPy is importable.

    Returns:
        True if CuPy can be imported, False otherwise.
    """
    global _cupy_available
    if _cupy_available is None:
        try:
            import cupy  # pragma: no cover
            _cupy_available = True  # pragma: no cover
        except ImportError:
            _cupy_available = False
    return _cupy_available


def get_cupy() -> types.ModuleType:
    """Returns the CuPy module, importing it lazily.

    Returns:
        The cupy module.

    Raises:
        ImportError: If CuPy is not installed.
    """
    global _cupy
    if _cupy is None:
        if not is_cupy_available():
            raise ImportError(
                "CuPy is required for CUDA support. "
                "Install it with: pip install pyecsago[cuda]"
            )
        import cupy  # pragma: no cover
        _cupy = cupy  # pragma: no cover
    return _cupy  # pragma: no cover
