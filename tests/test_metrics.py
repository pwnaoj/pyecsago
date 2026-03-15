"""Tests for pyecsago.utils.metrics."""

from pyecsago.utils.metrics import euclidean, minkowski, cosine, jaccard


class TestMetricsImports:
    def test_euclidean(self):
        assert euclidean([0, 0], [3, 4]) == 5.0

    def test_minkowski(self):
        result = minkowski([0, 0], [3, 4], p=2)
        assert result == 5.0

    def test_cosine(self):
        result = cosine([1, 0], [0, 1])
        assert result == 1.0  # orthogonal vectors

    def test_jaccard(self):
        result = jaccard([1, 0, 1], [1, 1, 0])
        assert 0 <= result <= 1
