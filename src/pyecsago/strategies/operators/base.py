"""Base evolutionary operator interface."""

from abc import ABC, abstractmethod


class EvolutionaryOperator(ABC):
    """Base interface for evolutionary operators.

    Defines how genetic operators work, including operator selection,
    application, and rate adjustment.
    """

    @abstractmethod
    def select_operator(self, operator_rates: dict) -> str:
        """Selects an operator based on the provided rates.

        Args:
            operator_rates: Dictionary mapping each operator to its rate.

        Returns:
            Name of the selected operator.
        """
        pass

    @abstractmethod
    def apply_operator(self, operator: str, parent1, parent2=None) -> list:
        """Applies the selected operator to the parents.

        Args:
            operator: Name of the operator to apply.
            parent1: First parent.
            parent2: Second parent (optional, for binary operators).

        Returns:
            List of generated offspring.
        """
        pass

    @abstractmethod
    def adjust_rates(self, individual, operator: str, reward: bool, delta: float):
        """Adjusts operator rates based on performance.

        Args:
            individual: Individual whose rates will be adjusted.
            operator: Operator used.
            reward: Whether to reward or penalize the operator.
            delta: Learning factor.
        """
        pass
