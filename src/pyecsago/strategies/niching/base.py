"""Base niching strategy interface."""

from abc import ABC, abstractmethod


class NichingStrategy(ABC):
    """Base interface for niching strategies.

    Defines how niches are managed in the population, including
    parent selection and individual replacement.
    """

    @abstractmethod
    def select_parents(self, individual: object, population: list, operator: str) -> list:
        """Selects parents for reproduction according to the niching strategy.

        Args:
            individual: Current individual needing parents for reproduction.
            population: Full population to select from.
            operator: Genetic operator to apply, determines how many parents are needed.

        Returns:
            List of individuals selected as parents.
        """
        pass

    def replace(self, parents: list, offspring: list) -> list:
        """Replacement policy for maintaining niche structure.

        In ECSAGO, replacement is done inline in the evolution loop
        (DC-HAEA Best*). This default implementation is a no-op.
        Subclasses may override if replacement logic is needed.
        """
        return offspring
