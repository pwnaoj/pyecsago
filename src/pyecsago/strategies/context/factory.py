"""strategies/context/factory.py"""

from __future__ import annotations

from typing import Any

from ...core.context import AlgorithmContext
from ...implementations.ecsago.context import StandardAlgorithmContext


class AlgorithmContextFactory:
    """
    Fábrica para crear contextos de algoritmo.

    Esta clase implementa el patrón Factory para crear
    la implementación adecuada del contexto según la configuración.
    """

    @staticmethod
    def create_context(config: dict[str, Any]) -> AlgorithmContext:
        """
        Crea un nuevo contexto de algoritmo basado en la configuración.
        
        Args:
            config: Configuración para el contexto
            
        Returns:
            Contexto de algoritmo configurado
        """
        use_cuda = config.get('use_cuda', False)
        
        # Crear instancia de contexto estándar
        return StandardAlgorithmContext(use_cuda=use_cuda)
