# read version from installed package
from importlib.metadata import version
__version__ = version("pyecsago")

# import features
from .implementations.ecsago.algorithm import ECSAGO
from .core.algorithm import EvolutionaryAlgorithm
from .strategies.evolution.factory import ECSAGOStrategyFactory
from .strategies.evolution.ecsago import ECSAGOStrategy
from .implementations.ecsago.population import ECSAGOPopulation
from .implementations.ecsago.individual import ECSAGOIndividual
from .strategies.fitness.ecsago import ECSAGOFitnessCalculator, CUDAECSAGOFitnessCalculator
from .strategies.operators.crossover import LinearCrossoverPerDimension
from .strategies.operators.mutation import AdaptiveGaussianMutation, GaussianMutation
from .strategies.niching.deterministic_crowding import DeterministicCrowding
from .strategies.operators.haea import HAEA
__all__ = [
    'ECSAGO',
    'EvolutionaryAlgorithm',
    'ECSAGOStrategyFactory',
    'ECSAGOStrategy',
    'ECSAGOPopulation',
    'ECSAGOIndividual',
    'ECSAGOFitnessCalculator',
    'CUDAECSAGOFitnessCalculator',
    'LinearCrossoverPerDimension',
    'AdaptiveGaussianMutation',
    'GaussianMutation',
    'DeterministicCrowding',
    'HAEA'
]
