"""Custom exception hierarchy for PyECSAGO."""


class PyECSAGOError(Exception):
    """Base exception for the PyECSAGO project."""
    pass


class ValidationError(PyECSAGOError, ValueError):
    """Raised when data or parameter validation fails."""
    pass


class ConfigurationError(PyECSAGOError, ValueError):
    """Raised when algorithm configuration is invalid."""
    pass


class EvolutionError(PyECSAGOError, RuntimeError):
    """Raised when an error occurs during the evolutionary process."""
    pass
