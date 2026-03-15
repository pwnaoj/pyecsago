"""Deterministic Crowding niching strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import NichingStrategy


if TYPE_CHECKING:
    from pyecsago import ECSAGOIndividual
    from pyecsago import HAEA
    from pyecsago.core.context import AlgorithmContext


class DeterministicCrowding(NichingStrategy):
    """Niching strategy via Deterministic Crowding.

    Maintains population diversity through competition between parents and
    their most similar offspring, preserving niches in the search space.
    """

    def __init__(self, operators_strategy: HAEA, context: AlgorithmContext) -> None:
        """Initializes the Deterministic Crowding strategy.

        Args:
            operators_strategy: Strategy managing genetic operators.
            context: Algorithm context (CPU/GPU).
        """
        self.operators_strategy = operators_strategy
        self.context = context

    def select_parents(
        self,
        individual: ECSAGOIndividual,
        population: list[ECSAGOIndividual],
        operator: str,
    ) -> list[ECSAGOIndividual]:
        """Selects parents based on ECSAGO strategy.

        For unary operators, returns only the current individual.
        For binary operators, selects a random second parent from the population
        with replacement (ECSAGO eliminates HAEA's mating restriction).

        Args:
            individual: Base individual for selection.
            population: Full population.
            operator: Operator type to apply.
        """
        arity = self.operators_strategy.get_operator_arity(operator)

        if arity is None:
            raise ValueError(f"Operator {operator} has no defined arity")

        if arity == 1:
            return [individual]

        elif arity == 2:
            parents = [individual]

            # ECSAGO: "the second individual is selected from the population
            # with replacement" (Dissertation Sec 3.3.3)
            candidates = [ind for ind in population if ind != individual]
            second_parent = self.context.dtype_strategy.random_choice(candidates, size=1, replace=True)[0]
            parents.append(second_parent)

            return parents
