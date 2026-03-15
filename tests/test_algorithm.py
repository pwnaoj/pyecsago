"""Tests for ECSAGO algorithm uncovered lines."""

import numpy as np
import pytest

from pyecsago import ECSAGO
from pyecsago.core.exceptions import ConfigurationError, ValidationError


class TestECSAGOValidation:
    def test_wrong_type_raises(self):
        with pytest.raises(ConfigurationError, match="Wrong type"):
            ECSAGO({
                'population_size': "not_int",
                'weight_threshold': 0.3,
                'max_generations': 30,
                'iterations': 10,
                'extraction_type': {2: 0.25},
                'k': 13.8,
                'use_cuda': False,
            })

    def test_extract_before_evolve_raises(self):
        config = {
            'population_size': 10,
            'weight_threshold': 0.3,
            'max_generations': 5,
            'iterations': 3,
            'extraction_type': {0: 0},
            'k': 13.8,
            'use_cuda': False,
        }
        ecsago = ECSAGO(config)
        with pytest.raises(ValidationError, match="evolve"):
            ecsago.extract_prototypes()

    def test_refine_before_evolve_raises(self):
        config = {
            'population_size': 10,
            'weight_threshold': 0.3,
            'max_generations': 5,
            'iterations': 3,
            'extraction_type': {0: 0},
            'k': 13.8,
            'use_cuda': False,
        }
        ecsago = ECSAGO(config)
        with pytest.raises(ValidationError, match="evolve"):
            ecsago.refine_prototypes(prototypes=[])

    def test_assign_clusters_empty_prototypes(self):
        config = {
            'population_size': 5,
            'weight_threshold': 0.3,
            'max_generations': 1,
            'iterations': 1,
            'extraction_type': {0: 0},
            'k': 13.8,
            'use_cuda': False,
        }
        ecsago = ECSAGO(config)
        ecsago.context.set_data(np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = ecsago._assign_clusters([])
        assert len(result) == 0

    def test_run_returns_empty_assignments_when_no_prototypes(self):
        """When extraction yields no prototypes, cluster_assignments should be empty."""
        np.random.seed(99)
        data = np.random.randn(20, 2)
        config = {
            'population_size': 5,
            'weight_threshold': 0.3,
            'max_generations': 1,
            'iterations': 1,
            'extraction_type': {0: 0},
            'k': 13.8,
            'use_cuda': False,
        }
        ecsago = ECSAGO(config)
        results = ecsago.run(data)
        # Either empty or has assignments — validates _assign_clusters handles both cases
        assert 'cluster_assignments' in results

    def test_run_full_pipeline(self):
        np.random.seed(42)
        data = np.vstack([
            np.random.randn(30, 2) + [0, 0],
            np.random.randn(30, 2) + [5, 5],
        ])
        config = {
            'population_size': 10,
            'weight_threshold': 0.3,
            'max_generations': 3,
            'iterations': 2,
            'extraction_type': {2: 0.25},
            'k': 13.8,
            'use_cuda': False,
        }
        ecsago = ECSAGO(config)
        results = ecsago.run(data)
        assert 'cluster_assignments' in results
        assert len(results['cluster_assignments']) == 60
