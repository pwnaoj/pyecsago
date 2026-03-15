"""strategies/refinement/factory.py"""

from .base import Refinement
from .mde import MDE
from .cuda_mde import CUDAMDE


class RefinementStrategyFactory:
    """
    Factory para crear estrategias de refinamiento configuradas.
    
    Esta clase simplifica la creación y configuración de estrategias
    de refinamiento, siguiendo el patrón Factory.
    """
    
    @staticmethod
    def create_mde(weight_threshold: float, sigma_factor: float, cuda_context: object | None, use_cuda: bool) -> Refinement:
        """
        Crea una estrategia de refinamiento MDE.
        
        Args:
            weight_threshold: Umbral para binarización de pesos
            sigma_factor: Factor K para determinar distancia mínima entre prototipos
            cuda_context: Contexto CUDA para operaciones en GPU
            use_cuda: Indica si se debe usar la implementación CUDA
            
        Returns:
            Estrategia de refinamiento MDE configurada
        """
        if use_cuda and cuda_context is not None:  # pragma: no cover
            return CUDAMDE(weight_threshold, sigma_factor)
        else:
            return MDE(weight_threshold, sigma_factor)
    