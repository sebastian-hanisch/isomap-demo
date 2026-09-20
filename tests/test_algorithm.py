import numpy as np
import pytest
from scipy.sparse.csgraph import shortest_path as sp_shortest_path
from sklearn.manifold import Isomap

from iso_algorithm import (
    classical_mds, connected_components, fit_isomap, knn_graph, pairwise_distances, residual_variance, shortest_path_route,
    shortest_paths, standardize,
)
from iso_scenario import generate_dataset


def _cloud(seed=0, n=90, d=5):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 3, n)
    return np.column_stack([np.cos(t), np.sin(t), t, 0.05 * rng.standard_normal((n, d - 3)).T]) if False else np.column_stack(
        [np.cos(t), np.sin(t), t] + [0.02 * rng.standard_normal(n) for _ in range(d - 3)])


def test_pairwise_distances_match_definition():
    Z = np.random.default_rng(0).standard_normal((30, 4))
    D = pairwise_distances(Z)
    assert np.allclose(D, np.linalg.norm(Z[:, None] - Z[None], axis=-1), atol=1e-6) and (np.diag(D) == 0).all() and (D == D.T).all()


def test_knn_graph_is_symmetric_and_has_at_least_k_neighbours_each():
    Z = np.random.default_rng(1).standard_normal((60, 3))
    weights, edges = knn_graph(pairwise_distances(Z), 4)
    finite = np.isfinite(weights)
    assert (finite == finite.T).all() and (finite.sum(1) - 1 >= 4).all()
    assert (edges[:, 0] < edges[:, 1]).all() and len(edges) == (finite.sum() - len(Z)) // 2


def test_floyd_warshall_matches_scipy_and_a_hand_graph():
    Z = np.random.default_rng(2).standard_normal((50, 3))
    weights, _ = knn_graph(pairwise_distances(Z), 3)
    ours = shortest_paths(weights)
    graph = np.where(np.isfinite(weights), weights, 0.0)
    assert np.allclose(np.where(np.isfinite(ours), ours, -1), np.where(np.isfinite(sp_shortest_path(graph, method="D", directed=False)),
                                                                        sp_shortest_path(graph, method="D", directed=False), -1), atol=1e-9)
    inf = np.inf
    hand = np.array([[0, 1, inf, inf], [1, 0, 2, inf], [inf, 2, 0, 3], [inf, inf, 3, 0]], dtype=float)     # Kette 0-1-2-3
    assert np.allclose(shortest_paths(hand)[0], [0, 1, 3, 6]) and shortest_paths(hand)[3, 0] == 6


def test_geodesics_are_symmetric_and_obey_the_triangle_inequality():
    Z = np.random.default_rng(3).standard_normal((40, 4))
    weights, _ = knn_graph(pairwise_distances(Z), 5)
    D = shortest_paths(weights)
    assert np.allclose(D, D.T)
    for a, b, c in np.random.default_rng(0).integers(0, 40, size=(200, 3)):
        assert D[a, c] <= D[a, b] + D[b, c] + 1e-9
    assert (D >= pairwise_distances(Z) - 1e-6).all()                                           # Weg nie kürzer als die Luftlinie


def test_shortest_path_route_has_the_reported_length():
    Z = np.random.default_rng(4).standard_normal((60, 3))
    weights, _ = knn_graph(pairwise_distances(Z), 4)
    D = shortest_paths(weights)
    route = shortest_path_route(weights, D, 0, 37)
    assert route[0] == 0 and route[-1] == 37
    assert abs(sum(weights[a, b] for a, b in zip(route[:-1], route[1:])) - D[0, 37]) < 1e-9


def test_connected_components_on_a_two_component_graph():
    inf = np.inf
    w = np.array([[0, 1, inf, inf, inf], [1, 0, inf, inf, inf], [inf, inf, 0, 1, 2], [inf, inf, 1, 0, 1], [inf, inf, 2, 1, 0]], dtype=float)
    labels = connected_components(np.isfinite(shortest_paths(w)))
    assert list(labels) == [1, 1, 0, 0, 0]                                                    # größte Komponente hat Label 0


def test_classical_mds_reproduces_euclidean_distances_exactly():
    Z = np.random.default_rng(5).standard_normal((40, 3))
    D = pairwise_distances(Z)
    Y, values = classical_mds(D, 3)
    assert np.allclose(pairwise_distances(Y), D, atol=1e-6)
    assert abs(values[3]) < 1e-6 * values[0]                                                   # Rang 3: der Rest ist Null


def test_isomap_matches_sklearn_for_the_same_neighbour_graph():
    X = _cloud(6)
    ours = fit_isomap(X, 8, 2)
    reference = Isomap(n_neighbors=8, n_components=2).fit(standardize(X))
    assert ours.connected
    assert np.allclose(ours.geodesic, reference.dist_matrix_, atol=1e-8)                      # geodätische Abstände identisch
    a, b = ours.embedding, reference.embedding_
    a, b = a - a.mean(0), b - b.mean(0)
    assert np.allclose(pairwise_distances(a), pairwise_distances(b), atol=1e-6)               # gleiche Einbettung bis auf Drehung/Spiegelung


def test_disconnected_graph_embeds_only_the_largest_component_and_reports_the_rest():
    ds = generate_dataset(300, 2, 1.0, 0.25, 0, 7)
    res = fit_isomap(ds.X, 2)
    assert not res.connected and res.n_dropped == 73 and len(res.indices) == 227
    assert res.embedding.shape[0] == len(res.indices) and int(res.components.max()) + 1 == 8


def test_residual_variance_drops_sharply_at_the_true_dimension():
    ds = generate_dataset(300, 2, 1.0, 0.25, 0, 7)
    res = fit_isomap(ds.X, 8)
    rv = residual_variance(res.geodesic, res.embedding, 6)
    assert rv[1] < rv[0] and rv[1] < 0.03 and rv[0] > 0.3
