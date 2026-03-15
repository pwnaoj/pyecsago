"""Tests for pyecsago.strategies.refinement."""

import numpy as np
import pytest

from pyecsago.strategies.refinement.mde import MDE
from pyecsago.strategies.refinement.base import Refinement


class TestMDE:
    def test_iteration_empty_prototypes(self, numpy_strategy):
        mde = MDE(weight_threshold=0.3, sigma_factor=13.8)
        result = mde.iteration([], np.array([[1, 2]]), numpy_strategy)
        assert result == []

    def test_single_iteration(self, sample_population, cpu_context, numpy_strategy):
        mde = MDE(weight_threshold=0.3)
        prototypes = sample_population[:3]
        data = cpu_context.get_data()
        result = mde.iteration(prototypes, data, numpy_strategy)
        assert len(result) == 3
        # Prototypes should be clones, not originals
        for r, p in zip(result, prototypes):
            assert r is not p

    def test_apply_multiple_iterations(self, sample_population, cpu_context, numpy_strategy):
        mde = MDE(weight_threshold=0.3)
        prototypes = sample_population[:3]
        data = cpu_context.get_data()
        result = mde.apply(prototypes, data, numpy_strategy, iterations=5)
        assert len(result) == 3

    def test_prototypes_converge(self, sample_population, cpu_context, numpy_strategy):
        mde = MDE(weight_threshold=0.3)
        prototypes = sample_population[:3]
        data = cpu_context.get_data()

        result1 = mde.apply(prototypes, data, numpy_strategy, iterations=1)
        result10 = mde.apply(prototypes, data, numpy_strategy, iterations=10)

        # After more iterations, prototypes should have moved
        genome1 = np.array([r.genome for r in result1])
        genome10 = np.array([r.genome for r in result10])
        assert not np.allclose(genome1, genome10)

    def test_winner_takes_all_assignment(self, sample_population, cpu_context, numpy_strategy):
        mde = MDE(weight_threshold=0.3)
        prototypes = sample_population[:3]
        data = cpu_context.get_data()
        indices, dists_sq = mde._assign_vectors_winner_takes_all(prototypes, data, numpy_strategy)
        assert len(indices) == len(data)
        assert np.all(dists_sq >= 0)
        assert np.all(indices < 3)

    def test_accumulate_stats(self, sample_population, cpu_context, numpy_strategy):
        mde = MDE(weight_threshold=0.3)
        prototypes = sample_population[:3]
        data = cpu_context.get_data()
        indices, dists_sq = mde._assign_vectors_winner_takes_all(prototypes, data, numpy_strategy)
        stats = mde._accumulate_stats(prototypes, data, indices, dists_sq, numpy_strategy)
        assert 'sum_weights' in stats
        assert 'sum_weights_feat' in stats
        assert 'sum_weights_dist_sq' in stats
        assert len(stats['sum_weights']) == 3


class TestRefinementBase:
    def test_init_noop(self):
        mde = MDE(weight_threshold=0.3)
        mde.init()  # should not raise

    def test_get_result_none(self):
        mde = MDE(weight_threshold=0.3)
        assert mde.get_result() is None
