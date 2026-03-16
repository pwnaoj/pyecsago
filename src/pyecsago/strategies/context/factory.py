"""strategies/context/factory.py"""

from __future__ import annotations

from typing import Any

from ...core.context import AlgorithmContext
from ...implementations.ecsago.context import StandardAlgorithmContext


class AlgorithmContextFactory:
    """Factory for creating algorithm context instances."""

    @staticmethod
    def create_context(config: dict[str, Any]) -> AlgorithmContext:
        """Creates a new algorithm context based on the configuration.

        Args:
            config: Algorithm configuration dictionary.

        Returns:
            Configured algorithm context.
        """
        use_cuda = config.get('use_cuda', False)
        return StandardAlgorithmContext(use_cuda=use_cuda)
