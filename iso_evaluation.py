"""Auswertung: findet Isomap die wahren Faktoren zurück - und wo bricht es? Alle Kennzahlen werden am Datensatz gemessen; die wahren latenten
Faktoren z sind bekannt (dank der Lieferrouten-Erzeugung).

- **Residualvarianz** (Tenenbaum et al. 2000): 1 − R² zwischen geodätischen Abständen und den Abständen der ersten d Koordinaten - Isomaps "Scree-Plot".
- **R² der wahren Faktoren**: wie gut lassen sich die latenten Faktoren aus den ersten zwei Koordinaten rekonstruieren (quadratische Regression, damit
  monotone Umparametrisierungen - z. B. Bogenlänge statt Faktor - nicht bestraft werden). PCA als Vergleich mit derselben Messung.
- **Kurzschluss-Anteil**: Anteil der Graphkanten, deren latente Länge mehr als das Dreifache dessen beträgt, was ihre Länge im Merkmalsraum bei
  lokalem Maßstab erwarten lässt (Maßstab = Median von latent/euklidisch über die Kanten des 3-NN-Graphen). Nur messbar, weil die Wahrheit bekannt ist."""

import time
from dataclasses import dataclass

import numpy as np

import iso_constants as C
from iso_algorithm import fit_isomap, knn_graph, pairwise_distances, residual_variance, standardize
from iso_scenario import generate_dataset

SHORT_FACTOR = 3.0
SHORT_MIN_SHARE = 0.002                # ab diesem Kurzschluss-Anteil (zusammen mit hoher Residualvarianz) warnt das Verdict
RV_TARGET = 0.05                       # "Knick": kleinste Dimension, deren Residualvarianz darunter liegt


def trustworthiness(X_high, X_low, n_neighbors=C.TRUST_NEIGHBORS):
    """Trustworthiness (Venna & Kaski, 2001): Anteil der Nachbarn im Einbettungsraum, die auch im Originalraum echte Nachbarn sind, mit
    Rang-Strafe für eingeschleppte Fremde. 1 = perfekt. Eigene Implementierung, gegen sklearn geprüft (nur im Test)."""
    n = len(X_high)
    k = n_neighbors
    d_high = np.linalg.norm(X_high[:, None, :] - X_high[None, :, :], axis=-1)
    d_low = np.linalg.norm(X_low[:, None, :] - X_low[None, :, :], axis=-1)
    np.fill_diagonal(d_high, np.inf)
    np.fill_diagonal(d_low, np.inf)
    ranks_high = np.argsort(np.argsort(d_high, axis=1), axis=1) + 1            # Rang 1 = nächster Nachbar
    neighbors_low = np.argsort(d_low, axis=1)[:, :k]
    penalty = 0.0
    for i in range(n):
        r = ranks_high[i, neighbors_low[i]]
        penalty += float(np.maximum(r - k, 0).sum())
    return 1.0 - 2.0 / (n * k * (2 * n - 3 * k - 1)) * penalty


def r2_quadratic(coords2, z):
    """R² der Rekonstruktion von z aus zwei Koordinaten (quadratische Regression, Mittel über die Faktoren, gewichtet mit ihrer Varianz)."""
    e1, e2 = coords2[:, 0], coords2[:, 1]
    A = np.column_stack([e1, e2, e1 ** 2, e1 * e2, e2 ** 2, np.ones(len(e1))])
    beta, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(1.0 - (z - A @ beta).var(0).sum() / z.var(0).sum())


def shortcut_share(euclid, edges, z, base_k=3):
    latent = pairwise_distances(z)
    _, base_edges = knn_graph(euclid, base_k)
    scale = float(np.median(latent[base_edges[:, 0], base_edges[:, 1]] / np.maximum(euclid[base_edges[:, 0], base_edges[:, 1]], 1e-12)))
    predicted = scale * euclid[edges[:, 0], edges[:, 1]]
    return float((latent[edges[:, 0], edges[:, 1]] > SHORT_FACTOR * predicted).mean())


def pca_project(X, n_components=2):
    Z = standardize(X)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ vt[:n_components].T


def dimension_elbow(rv, target=RV_TARGET):
    """Kleinste Dimension mit Residualvarianz <= target (None, wenn keine)."""
    hits = np.nonzero(np.asarray(rv) <= target)[0]
    return int(hits[0]) + 1 if len(hits) else None


@dataclass(frozen=True)
class Analysis:
    result: object
    rv: np.ndarray                   # Residualvarianz je Dimension 1..N_COMPONENTS_MAX
    q_hat: object                    # Isomap-Dimensionsschätzung (Knick) oder None
    r2_iso: float
    r2_pca: float
    trust_iso: float
    trust_pca: float
    short_share: float
    pca_2d: np.ndarray
    iso_2d: np.ndarray
    indices: np.ndarray


def analyse(dataset, k):
    res = fit_isomap(dataset.X, k, C.N_COMPONENTS_MAX)
    rv = residual_variance(res.geodesic, res.embedding, res.embedding.shape[1])
    z = dataset.z[res.indices]
    Z = standardize(dataset.X)
    pca2 = pca_project(dataset.X)
    iso2 = res.embedding[:, :2]
    return Analysis(
        result=res, rv=rv, q_hat=dimension_elbow(rv), r2_iso=r2_quadratic(iso2, z), r2_pca=r2_quadratic(pca2, dataset.z),
        trust_iso=trustworthiness(Z[res.indices], iso2), trust_pca=trustworthiness(Z, pca2), short_share=shortcut_share(res.euclid, res.edges, dataset.z),
        pca_2d=pca2, iso_2d=iso2, indices=res.indices,
    )


def verdict(analysis, dataset, k):
    """Verdict-Kaskade (Warnungen zuerst) -> (Stufe, Code, Daten)."""
    a, res = analysis, analysis.result
    data = {"k": k, "q": dataset.q, "dropped": res.n_dropped, "components": int(res.components.max()) + 1, "short": a.short_share, "q_hat": a.q_hat,
            "rv_q": float(a.rv[min(dataset.q, len(a.rv)) - 1]), "r2_iso": a.r2_iso, "r2_pca": a.r2_pca, "trust_iso": a.trust_iso, "trust_pca": a.trust_pca}
    if not res.connected:
        return "warning", "disconnected", data
    if a.short_share >= SHORT_MIN_SHARE and data["rv_q"] > 0.1:
        return "warning", ("noise_short_circuit" if dataset.noise >= 0.4 else "short_circuit"), data
    if dataset.curvature == 0 and a.r2_iso - a.r2_pca < 0.03:
        return "info", "no_advantage", data
    if a.r2_iso - a.r2_pca >= 0.10 and data["rv_q"] <= 0.1:
        return "success", "isomap_wins", data
    return "info", "neutral", data


def make_dataset(n_tours, q, curvature, noise, seed):
    return generate_dataset(n_tours, q, curvature, noise, 0, seed)


def k_sweep(q, curvature, noise, n_tours=C.SWEEP_N_TOURS, ks=C.SWEEP_KS, seeds=C.SWEEP_SEEDS):
    """Feste Sweep-Seeds (unabhängig vom Demo-Seed): je k Anteil zusammenhängender Graphen, mittlere Residualvarianz bei q, R², Kurzschluss-Anteil."""
    rows = []
    for k in ks:
        conn, rvq, r2, short = [], [], [], []
        for seed in seeds:
            ds = make_dataset(n_tours, q, curvature, noise, seed)
            res = fit_isomap(ds.X, k, max(q, 2))
            conn.append(res.connected)
            rv = residual_variance(res.geodesic, res.embedding, res.embedding.shape[1])
            rvq.append(float(rv[min(q, len(rv)) - 1]))
            r2.append(r2_quadratic(res.embedding[:, :2], ds.z[res.indices]))
            short.append(shortcut_share(res.euclid, res.edges, ds.z))
        rows.append({"k": int(k), "connected": float(np.mean(conn)), "rv_q": float(np.mean(rvq)), "r2": float(np.mean(r2)), "short": float(np.mean(short))})
    return rows


def timing_sweep(ns=C.TIMING_NS, k=C.TIMING_K, q=2, seed=100_000):
    """Gemessene Rechenzeit (Sekunden) von Isomap und PCA für wachsende n (eigene Messung, Rechner-abhängig)."""
    rows = []
    for n in ns:
        ds = generate_dataset(n, q, 0.0, 0.1, 0, seed)
        t = time.perf_counter()
        fit_isomap(ds.X, k, 2)
        t_iso = time.perf_counter() - t
        t = time.perf_counter()
        pca_project(ds.X)
        t_pca = time.perf_counter() - t
        rows.append({"n": int(n), "isomap": t_iso, "pca": t_pca})
    return rows
