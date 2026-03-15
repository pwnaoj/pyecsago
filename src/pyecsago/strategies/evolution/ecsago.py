"""ECSAGO evolution strategy implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EvolutionStrategy


if TYPE_CHECKING:
    from pyecsago import DeterministicCrowding, HAEA, ECSAGOIndividual, ECSAGOFitnessCalculator, CUDAECSAGOFitnessCalculator
    from pyecsago.core.context import AlgorithmContext


class ECSAGOStrategy(EvolutionStrategy):
    """ECSAGO-specific evolution strategy.

    Combines adaptive operators via HAEA with niche maintenance
    via Deterministic Crowding. Supports both CPU (NumPy) and GPU
    (CuPy/CUDA) transparently.
    """

    def __init__(self,
                 niching_strategy: DeterministicCrowding,
                 operators_strategy: HAEA,
                 weight_threshold: float,
                 fitness_calculator: ECSAGOFitnessCalculator | CUDAECSAGOFitnessCalculator,
                 context: AlgorithmContext,
                ) -> None:
        """Initializes the ECSAGO strategy with its core components.

        Args:
            niching_strategy: Strategy for maintaining population niches.
            operators_strategy: Adaptive operator strategy.
            weight_threshold: Threshold for weight binarization.
            fitness_calculator: Fitness calculator (CPU or CUDA).
            context: Algorithm context (manages CPU/GPU).
        """
        self.niching_strategy = niching_strategy
        self.operators_strategy = operators_strategy
        self.weight_threshold = weight_threshold
        self.fitness_calculator = fitness_calculator
        self.context = context

    def _evaluate_fitness(self, offsprings: list[ECSAGOIndividual], individual: ECSAGOIndividual) -> None:
        """Evaluates fitness of offspring and the current individual.

        In CPU mode, calculates fitness individually.
        In CUDA mode, uses calculate_with_kernels for batch processing.
        """
        if hasattr(self.fitness_calculator, 'calculate_with_kernels'):  # pragma: no cover
            # CUDA path: batch evaluation
            offsprings.append(individual)
            self.fitness_calculator.calculate_with_kernels(offsprings, self.weight_threshold)
        else:
            # CPU path: individual evaluation
            for offspring in offsprings:
                offspring.calculate_fitness(self.weight_threshold)
            individual.calculate_fitness(self.weight_threshold)
            offsprings.append(individual)

    def evolve_population(self, population: list[ECSAGOIndividual]) -> list[ECSAGOIndividual]:
        """Evolves the population following the ECSAGO algorithm.

        Performs a complete evolution cycle:
        1. Selects an operator for each individual.
        2. Selects parents according to the niching strategy.
        3. Applies the selected operators.
        4. Updates the population according to the replacement policy.

        Args:
            population: Current population to evolve.

        Returns:
            New evolved population.
        """
        def _evolve():
            new_population = []
            new_rates = []

            for individual in population:
                # Select operator according to its rates
                current_operators = individual.operator_rates.copy()
                operator = self.operators_strategy.select_operator(current_operators)

                # Select parents based on operator arity
                parents = self.niching_strategy.select_parents(
                    individual,
                    population,
                    operator
                )

                # Generate offspring
                offsprings = self.operators_strategy.apply_operator(
                    self.context.dtype_strategy,
                    operator,
                    parents[0],
                    parents[1] if len(parents) > 1 else None
                )

                # Evaluate fitness (CPU or CUDA depending on fitness_calculator)
                self._evaluate_fitness(offsprings, individual)

                # Select best child according to DC-HAEA
                best_child = self.fitness_calculator._select_best_offspring(individual, offsprings)

                # Copy rates for adjustment
                new_operators = current_operators.copy()

                # Adjust operator rates
                if best_child.fitness > individual.fitness:
                    new_operators = self.operators_strategy.adjust_rates(new_operators, operator, reward=True)
                else:
                    new_operators = self.operators_strategy.adjust_rates(new_operators, operator, reward=False)

                # Normalize the new rates
                rates_array = self.context.dtype_strategy.array(list(new_operators.values()))
                normalized_rates = rates_array / self.context.dtype_strategy.module.sum(rates_array)
                new_operators = dict(zip(new_operators.keys(), normalized_rates))

                # Add best child to the new population
                new_population.append(best_child)
                new_rates.append(new_operators)

            # Update rates for all individuals at the end of the evolution cycle
            for i, ind in enumerate(new_population):
                ind.operator_rates = new_rates[i]

            return new_population

        # Execute in the appropriate context (direct CPU or CUDA wrapping)
        return self.context.execute(_evolve)
