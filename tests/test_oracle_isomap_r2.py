"""Orakel-Test für `r2_quadratic`: sklearn (PolynomialFeatures + LinearRegression + r2_score, varianzgewichtet) auf Zufallsfällen mit konstantem Versatz,
Skalenunterschied und entarteter Koordinate. Im Training enthält die Regression einen Achsenabschnitt, die Residuen sind dort also mittelwertfrei;
die Formel stimmt deshalb mit R² = 1 − SSE/SST überein (anders als bei zurückgehaltenen Daten, wo ein Versatz zählen muss)."""

import numpy as np
import pytest

from iso_evaluation import r2_quadratic

sk_lm = pytest.importorskip("sklearn.linear_model")
sk_metrics = pytest.importorskip("sklearn.metrics")
sk_pre = pytest.importorskip("sklearn.preprocessing")


def test_r2_quadratic_equals_sklearn_r2_score_on_random_cases():
    rng = np.random.default_rng(5)
    for t in range(60):
        n, q = int(rng.integers(8, 70)), int(rng.integers(2, 5))
        coords = rng.normal(size=(n, 2)) * rng.uniform(0.2, 5, 2) + rng.normal(size=2) * 4
        z = rng.normal(size=(n, q)) * rng.uniform(0.2, 5, q) + rng.normal(size=q) * 3
        if t % 4 == 1:
            z = z + 50.0
        elif t % 4 == 2:
            z = 1e-3 * z
        elif t % 4 == 3:
            coords[:, 1] = coords[0, 1]                                  # entartete Koordinate: Rang-defizientes Modell
        poly = sk_pre.PolynomialFeatures(2).fit_transform(coords)
        pred = sk_lm.LinearRegression(fit_intercept=False).fit(poly, z).predict(poly)
        assert r2_quadratic(coords, z) == pytest.approx(sk_metrics.r2_score(z, pred, multioutput="variance_weighted"), abs=1e-6)
