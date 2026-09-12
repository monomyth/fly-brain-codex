import numpy as np
import pytest
from scipy import sparse
from fly_brain.connectome import map_ids, normalize_counts, io_reachability
from fly_brain.ablations import directed_swaps


def test_direction_and_normalization():
    counts = sparse.csr_matrix(([3, 1], ([1, 1], [0, 2])), shape=(3, 3))
    normalized = normalize_counts(counts)
    np.testing.assert_allclose(normalized @ [4, 0, 8], [0, 5, 0])
    np.testing.assert_allclose(np.asarray(normalized.sum(axis=1)).ravel(), [0, 1, 0])


def test_body_ids_do_not_round_or_alias():
    ids = np.array([1, 2**53 + 1, 2**53 + 3], dtype=np.int64)
    mapped, retained = map_ids(ids, np.array([1, 2**53 + 2, 2**53 + 3, 2**62], dtype=np.int64))
    assert retained.tolist() == [True, False, True, False]
    assert mapped[retained].tolist() == [0, 2]


def test_directed_reachability(graph):
    reachable, _ = io_reachability(graph.matrix, graph.inputs, graph.outputs)
    assert reachable.all()
    reverse, _ = io_reachability(graph.matrix, graph.outputs, graph.inputs)
    assert not reverse.any()


def test_rewiring_preserves_degrees_and_weights():
    rng = np.random.default_rng(22)
    data = rng.integers(1, 20, size=(20, 20)) * (rng.random((20, 20)) < .15)
    np.fill_diagonal(data, 0)
    counts = sparse.csr_matrix(data)
    rewired, report = directed_swaps(counts, seed=3, swaps_per_edge=2)
    assert report["accepted_swaps"] == report["requested_swaps"]
    assert (rewired != counts).nnz > 0
    assert sorted(rewired.data) == sorted(counts.data)
    np.testing.assert_array_equal(np.diff(rewired.indptr), np.diff(counts.indptr))
    np.testing.assert_array_equal(np.diff(rewired.tocsc().indptr), np.diff(counts.tocsc().indptr))
    np.testing.assert_array_equal(np.asarray(rewired.sum(axis=0)), np.asarray(counts.sum(axis=0)))
