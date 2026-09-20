"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem OR-Demo-Portfolio, siehe km_presets.py in kmeans-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import iso_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "n_tours_slider": SettingSpec("n", int, C.DEFAULT_N_TOURS, C.N_TOURS_MIN, C.N_TOURS_MAX),
    "q_slider": SettingSpec("q", int, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "curvature_slider": SettingSpec("curv", float, C.DEFAULT_CURVATURE, C.CURVATURE_MIN, C.CURVATURE_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(n_tours, q, curvature, noise, k, seed):
    try:
        st.query_params["n"] = str(int(n_tours))
        st.query_params["q"] = str(int(q))
        st.query_params["curv"] = str(curvature)
        st.query_params["noise"] = str(noise)
        st.query_params["k"] = str(int(k))
        st.query_params["seed"] = str(int(seed))
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    st.session_state["n_tours_slider"] = p["n_tours"]
    st.session_state["q_slider"] = p["q"]
    st.session_state["curvature_slider"] = p["curvature"]
    st.session_state["noise_slider"] = p["noise"]
    st.session_state["k_slider"] = p["k"]
    st.session_state["seed_input"] = p["seed"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
