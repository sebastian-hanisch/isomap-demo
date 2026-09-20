"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Schritt-Zustand, Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import iso_constants as C

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None):
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def test_default_renders_without_exception():
    at = _run()
    assert any("von k" in h.value for h in at.subheader)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    p = C.PRESETS[name]

    def setup(at):
        at.session_state["n_tours_slider"] = p["n_tours"]
        at.session_state["q_slider"] = p["q"]
        at.session_state["curvature_slider"] = p["curvature"]
        at.session_state["noise_slider"] = p["noise"]
        at.session_state["k_slider"] = p["k"]
        at.session_state["seed_input"] = p["seed"]
    _run(setup)


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MIN
        at.session_state["k_slider"] = C.K_MIN                       # k = 1: Graph in vielen Teilen
        at.session_state["q_slider"] = C.Q_MIN
    _run(small)

    def large(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MAX
        at.session_state["k_slider"] = C.K_MAX
        at.session_state["q_slider"] = C.Q_MAX
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["curvature_slider"] = C.CURVATURE_MAX
    _run(large)


def test_step_state_resets_when_the_data_changes_and_survives_reruns():
    at = _run()
    at.session_state["iso_step"] = 3
    at.run()
    assert not at.exception and at.session_state["iso_step"] == 3
    at.session_state["k_slider"] = 12
    at.run()
    assert not at.exception and at.session_state["iso_step"] == 1


@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_every_step_renders(step):
    def setup(at):
        at.session_state["iso_step"] = step
    _run(setup)


def test_every_figure_of_the_visualisation_module_is_axis_locked():
    source = (ROOT / "iso_visualization.py").read_text(encoding="utf-8")
    assert len(re.findall(r"return lock_axes\(fig\)", source)) == 6
    assert len(re.findall(r"^\s+return fig$", source, flags=re.M)) == 1                       # nur das return von lock_axes selbst
