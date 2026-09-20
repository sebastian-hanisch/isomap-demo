import numpy as np
import pytest
from sklearn.manifold import trustworthiness as sk_trustworthiness

import iso_constants as C
from iso_algorithm import fit_isomap, knn_graph, pairwise_distances, standardize
from iso_evaluation import (
    analyse, dimension_elbow, k_sweep, make_dataset, pca_project, r2_quadratic, shortcut_share, timing_sweep, trustworthiness, verdict,
)
from iso_scenario import generate_dataset


def test_scenario_is_bit_identical_to_the_pca_demo_generator():
    """Eingefrorene Referenzwerte aus pca-demo (`generate_dataset`, gleiche Argumente): die Kopie darf nicht abweichen."""
    d = generate_dataset(300, 2, 1.0, 0.25, 0, 7)
    assert d.X.shape == (300, 12) and abs(float(d.X.sum()) - 14553337.310875032) < 1e-6
    assert np.allclose(d.X[0, :3], [54469.13732407575, 74.83224937121233, 767.2459701758783]) and abs(float(d.z.sum()) - (-80.2378453142044)) < 1e-9
    assert abs(float(generate_dataset(200, 3, 0.4, 0.3, 5, 42).X.sum()) - 10763498.969287368) < 1e-6


def test_make_dataset_is_deterministic_and_seed_dependent():
    a, b = make_dataset(150, 2, 0.5, 0.2, 3), make_dataset(150, 2, 0.5, 0.2, 3)
    assert np.array_equal(a.X, b.X) and not np.array_equal(a.X, make_dataset(150, 2, 0.5, 0.2, 4).X)


def test_trustworthiness_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6))
    for embedding in (X[:, :2], rng.standard_normal((80, 2)), pca_project(X)):
        assert abs(trustworthiness(X, embedding, 8) - sk_trustworthiness(X, embedding, n_neighbors=8)) < 1e-9


def test_r2_quadratic_recovers_monotone_reparametrisations_and_rejects_noise():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((200, 2))
    coords = np.column_stack([z[:, 0] + 0.3 * z[:, 0] ** 2, z[:, 1] + 0.2 * z[:, 1] ** 2])       # glatte Umparametrisierung
    assert r2_quadratic(coords, z) > 0.9
    assert r2_quadratic(rng.standard_normal((200, 2)), z) < 0.1


def test_shortcut_share_is_zero_for_a_flat_sheet_and_positive_for_a_constructed_shortcut():
    rng = np.random.default_rng(2)
    z = rng.uniform(0, 4, (200, 2))
    flat_edges = knn_graph(pairwise_distances(z), 6)[1]
    assert shortcut_share(pairwise_distances(z), flat_edges, z) < 0.01
    # gefaltetes Blatt: dieselben latenten Punkte, aber die zweite Hälfte liegt im Merkmalsraum nahe über der ersten (ohne latent nahe zu sein)
    folded = np.where(np.arange(200)[:, None] < 100, z, z * [1.0, 1.0] + [0.0, 0.0])
    X = np.column_stack([folded, np.where(np.arange(200) < 100, 0.0, 0.05)])
    X[100:, :2] = z[:100] + 0.01 * rng.standard_normal((100, 2))                                # Punkt 100+i liegt bei Punkt i, aber latent woanders
    latent = z.copy()
    latent[100:] = z[100:]
    euclid = pairwise_distances(X)
    edges = knn_graph(euclid, 3)[1]
    assert shortcut_share(euclid, edges, latent) > 0.05


def test_dimension_elbow():
    assert dimension_elbow([0.5, 0.04, 0.03]) == 2 and dimension_elbow([0.5, 0.4]) is None and dimension_elbow([0.01, 0.0]) == 1


def test_isomap_beats_pca_on_curved_data_and_ties_on_flat_data():
    curved = analyse(make_dataset(300, 2, 1.0, 0.25, 7), 8)
    assert curved.r2_iso > 0.95 and curved.r2_pca < 0.6 and curved.trust_iso > curved.trust_pca + 0.1
    flat = analyse(make_dataset(300, 2, 0.0, 0.25, 7), 10)
    assert abs(flat.r2_iso - flat.r2_pca) < 0.05


def test_k_sweep_is_deterministic_and_shows_the_window_on_curved_data():
    rows = k_sweep(2, 1.0, 0.25, n_tours=200, ks=(2, 8, 45), seeds=C.SWEEP_SEEDS[:2])
    assert rows == k_sweep(2, 1.0, 0.25, n_tours=200, ks=(2, 8, 45), seeds=C.SWEEP_SEEDS[:2])
    small, good, large = rows
    assert small["connected"] < 1.0 and good["connected"] == 1.0
    assert good["rv_q"] < 0.05 < large["rv_q"] and good["r2"] > large["r2"] and large["short"] > good["short"]
    assert C.SWEEP_SEEDS[0] >= 100_000 > C.DEFAULT_SEED


def test_on_flat_data_a_large_k_does_not_hurt():
    rows = k_sweep(2, 0.0, 0.25, n_tours=200, ks=(8, 45), seeds=C.SWEEP_SEEDS[:2])
    assert rows[1]["rv_q"] <= rows[0]["rv_q"] + 0.01 and rows[1]["short"] < 0.005


def test_verdict_cascade_codes():
    def code(q, curvature, noise, k, seed=7):
        ds = make_dataset(300, q, curvature, noise, seed)
        return verdict(analyse(ds, k), ds, k)[1]
    assert code(2, 1.0, 0.25, 8) == "isomap_wins"
    assert code(2, 0.0, 0.25, 10) == "no_advantage"
    assert code(2, 1.0, 0.25, 2) == "disconnected"
    assert code(2, 1.0, 0.25, 50) == "short_circuit"
    assert code(2, 1.0, 0.8, 10) == "noise_short_circuit"
    assert code(3, 1.0, 0.25, 10) == "short_circuit"


def test_timing_sweep_is_monotone_and_isomap_is_slower_than_pca():
    rows = timing_sweep(ns=(100, 300))
    assert rows[1]["isomap"] > rows[0]["isomap"] and all(r["isomap"] > 10 * r["pca"] for r in rows)
