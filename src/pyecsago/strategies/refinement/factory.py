"""strategies/refinement/factory.py"""

from .base import Refinement
from .mde import MDE
from .cuda_mde import CUDAMDE


class RefinementStrategyFactory:
    """Factory for creating configured refinement strategies."""

    @staticmethod
    def create_mde(weight_threshold: float, sigma_factor: float, cuda_context: object | None, use_cuda: bool) -> Refinement:
        """Creates an MDE refinement strategy.

        Args:
            weight_threshold: Threshold for weight binarization.
            sigma_factor: K factor for minimum distance between prototypes.
            cuda_context: CUDA context for GPU operations.
            use_cuda: Whether to use the CUDA implementation.

        Returns:
            Configured MDE refinement strategy.
        """
        if use_cuda and cuda_context is not None:  # pragma: no cover
            return CUDAMDE(weight_threshold, sigma_factor)
        else:
            return MDE(weight_threshold, sigma_factor)
