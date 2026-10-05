"""Orakel-Tests (unabhängiger Rechenweg): Graph, Komponenten und geodätische Abstände per networkx/Dijkstra, Einbettung und Eigenwerte gegen sklearn auf
vielen Zufallsinstanzen, Residualvarianz und Kurzschluss-Anteil von Hand."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist, squareform

import iso_algorithm as M
import iso_evaluation as E
import iso_scenario as S

nx = pytest.importorskip("networkx")
manifold = pytest.importorskip("sklearn.manifold")


def _cloud(rng, kind):
    n = int(rng.integers(15, 60))
    if kind == 0:
        return rng.normal(size=(n, 4)) * rng.uniform(0.5, 4, size=4)
    if kind == 1:
        t = 1.5 * np.pi * (1 + 2 * rng.random(n))
        return np.stack([t * np.cos(t), 10 * rng.random(n), t * np.sin(t)], 1) + 0.3 * rng.normal(size=(n, 3))
    return np.vstack([rng.normal(size=(n // 2, 3)), rng.normal(size=(n - n // 2, 3)) + 50])      # zwei getrennte Wolken


def test_graph_components_and_geodesics_equal_networkx_dijkstra_and_connected_graphs_equal_sklearn_isomap():
    rng = np.random.default_rng(1)
    for t in range(45):
        X = _cloud(rng, t % 3)
        n, k = len(X), int(rng.integers(2, min(12, len(X) - 1)))
        Z = M.standardize(X)
        res = M.fit_isomap(X, k, 2)
        D = cdist(Z, Z)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in [j for j in np.argsort(D[i], kind="stable") if j != i][:k]:
                G.add_edge(i, int(j), weight=D[i, j])
        assert sorted((int(a), int(b)) for a, b in res.edges) == sorted((min(a, b), max(a, b)) for a, b in G.edges)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        assert int(res.components.max()) + 1 == len(comps)
        if len(comps) == 1 or len(comps[0]) > len(comps[1]):
            assert set(res.indices.tolist()) == set(comps[0])
        sp = dict(nx.all_pairs_dijkstra_path_length(G))
        ref = np.array([[sp[int(i)][int(j)] for j in res.indices] for i in res.indices])
        assert np.allclose(ref, res.geodesic, atol=1e-9)
        if len(comps) == 1:
            sk = manifold.Isomap(n_neighbors=k, n_components=2, eigen_solver="dense").fit(Z)
            a, b = res.embedding - res.embedding.mean(0), sk.embedding_ - sk.embedding_.mean(0)
            assert np.allclose(cdist(a, a), cdist(b, b), atol=1e-6 * max(1.0, cdist(b, b).max()))
            assert np.allclose(np.sort(res.eigenvalues)[::-1][:2], sk.kernel_pca_.eigenvalues_[:2], rtol=1e-6)


def test_residual_variance_equals_one_minus_squared_correlation_by_hand():
    X = S.generate_dataset(120, 2, 1.0, 0.25, 0, 7).X
    res = M.fit_isomap(X, 10, 5)
    rv = M.residual_variance(res.geodesic, res.embedding, 5)
    iu = np.triu_indices(len(res.geodesic), 1)
    for d in range(1, 6):
        r = np.corrcoef(res.geodesic[iu], cdist(res.embedding[:, :d], res.embedding[:, :d])[iu])[0, 1]
        assert rv[d - 1] == pytest.approx(1 - r ** 2, abs=1e-8)


def test_shortcut_share_equals_the_definition_by_hand():
    rng = np.random.default_rng(2)
    for _ in range(8):
        ds = S.generate_dataset(int(rng.integers(30, 70)), int(rng.integers(1, 4)), float(rng.uniform(0, 1)), float(rng.uniform(0, 1)), 0, int(rng.integers(10 ** 6)))
        res = M.fit_isomap(ds.X, int(rng.integers(3, 12)), 2)
        lat, euc = squareform(pdist(ds.z)), squareform(pdist(M.standardize(ds.X)))
        base = set()
        for i in range(len(euc)):
            for j in [j for j in np.argsort(euc[i], kind="stable") if j != i][:3]:
                base.add((min(i, j), max(i, j)))
        scale = np.median([lat[i, j] / max(euc[i, j], 1e-12) for i, j in base])
        expected = np.mean([lat[i, j] > 3.0 * scale * euc[i, j] for i, j in res.edges])
        assert E.shortcut_share(res.euclid, res.edges, ds.z) == pytest.approx(expected, abs=1e-12)


def test_pca_projection_equals_the_covariance_eigenvectors_up_to_sign_and_elbow_hand_cases():
    X = S.generate_dataset(100, 2, 1.0, 0.25, 0, 3).X
    Z = M.standardize(X)
    top = np.linalg.eigh(np.cov(Z.T))[1][:, ::-1][:, :2]
    assert np.allclose(np.abs(E.pca_project(X)), np.abs(Z @ top), atol=1e-8)
    assert E.dimension_elbow([0.5, 0.04, 0.01]) == 2 and E.dimension_elbow([0.5, 0.4]) is None and E.dimension_elbow([0.05]) == 1
