"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert)."""

import pytest

import iso_constants as C
from iso_evaluation import analyse, make_dataset, verdict


def _measure(p):
    dataset = make_dataset(p["n_tours"], p["q"], p["curvature"], p["noise"], p["seed"])
    a = analyse(dataset, p["k"])
    code, data = verdict(a, dataset, p["k"])[1:]
    return {"verdict": code, "dropped": data["dropped"], "components": data["components"], "short": data["short"], "q_hat": data["q_hat"],
            "rv_q": data["rv_q"], "r2_iso": data["r2_iso"], "r2_pca": data["r2_pca"], "trust_iso": data["trust_iso"]}


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_TOURS_MIN <= p["n_tours"] <= C.N_TOURS_MAX and C.Q_MIN <= p["q"] <= C.Q_MAX
        assert C.CURVATURE_MIN <= p["curvature"] <= C.CURVATURE_MAX and C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX and C.K_MIN <= p["k"] <= C.K_MAX


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if isinstance(expected, str):
            assert value == expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
