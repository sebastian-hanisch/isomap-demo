"""Isomap an Lieferrouten-Kennzahlen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Isomap - und lässt
stattdessen das Beispiel wachsen. Zweites Stück der Dimensionsreduktion-Linie der "Konzepte"-Reihe: Isomap behebt die Linearitätsschwäche der PCA
(pca-demo) über geodätische Abstände entlang eines Nachbarschaftsgraphen - und hat dafür eigene Schwächen (Kurzschlüsse, Zusammenhang, Rauschen,
kubische Rechenzeit). Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import iso_constants as C
from iso_algorithm import shortest_path_route
from iso_evaluation import RV_TARGET, analyse, k_sweep, make_dataset, timing_sweep, verdict
from iso_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from iso_visualization import (
    build_distance_scatter,
    build_embedding,
    build_graph_figure,
    build_k_sweep,
    build_residual_curve,
    build_timing,
)

st.set_page_config(page_title="Isomap – Sebastian Hanisch", layout="wide")

STEP_LABELS = {
    1: "1 · Nachbarschaftsgraph",
    2: "2 · Kürzester Weg",
    3: "3 · Luftlinie gegen Weg",
    4: "4 · Einbettung (MDS)",
}


@st.cache_data(show_spinner=False)
def _dataset(n_tours, q, curvature, noise, seed):
    return make_dataset(n_tours, q, curvature, noise, seed)


@st.cache_data(show_spinner=False)
def _analysis(n_tours, q, curvature, noise, seed, k):
    return analyse(make_dataset(n_tours, q, curvature, noise, seed), k)


@st.cache_data(show_spinner=False)
def _sweep(q, curvature, noise):
    return k_sweep(q, curvature, noise)


st.title("🗺️ Isomap an Lieferrouten-Kennzahlen")
st.markdown(
    """
Dieselben **12 Kennzahlen je Lieferroute** wie in der PCA-Demo - erzeugt aus wenigen versteckten Faktoren, aber mit **gekrümmter** Struktur, an der
die PCA scheiterte: gerade Achsen können eine gebogene Fläche nicht mit wenigen Koordinaten beschreiben. **Isomap** misst Entfernungen deshalb nicht in
der Luftlinie, sondern **entlang der Fläche**: Es verbindet jede Tour mit ihren *k* ähnlichsten Nachbarn, bestimmt die kürzesten Wege durch dieses Netz
(die **geodätischen Abstände**) und bettet die Touren so ein, dass diese Wege-Abstände erhalten bleiben. Genau **wie** das funktioniert, erklärt der
aufgeklappte Abschnitt direkt darunter - bevor weiter unten das Verfahren Schritt für Schritt läuft und die Frage "📐 Wie stark hängt das Ergebnis von k
ab?" live beantwortet wird.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - zweites Stück der "
    "Dimensionsreduktion-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel: Isomap behebt die Linearitätsschwäche der PCA "
    "(pca-demo), hat aber eigene: Kurzschlüsse bei zu großem k, einen zerfallenden Graphen bei zu kleinem k und kubische Rechenzeit."
)

with st.expander("So funktioniert Isomap", expanded=True):
    st.markdown(
        """
Isomap (Tenenbaum, de Silva & Langford, 2000) besteht aus drei Schritten:

1. **Nachbarschaftsgraph**: jede Tour wird mit ihren *k* nächsten Nachbarn (im Merkmalsraum, in z-Werten) verbunden. Die Kantenlänge ist der euklidische Abstand.
2. **Kürzeste Wege**: die Länge des kürzesten Weges durch den Graphen zwischen zwei Touren nähert ihren **geodätischen Abstand** an - die Strecke *auf* der
   Fläche statt der Luftlinie hindurch. Bei einer gebogenen Fläche ist sie länger als die Luftlinie; bei einer geraden gleich lang.
3. **Klassisches MDS**: aus der Matrix aller geodätischen Abstände werden Koordinaten berechnet, deren Abstände sie möglichst genau reproduzieren (Eigenzerlegung der
   doppelt zentrierten Abstandsmatrix). Die **Residualvarianz** zeigt, wie viele Dimensionen dafür nötig sind - ihr Knick ist Isomaps Schätzung der wahren Dimension.

Das funktioniert nur, wenn *k* stimmt: zu klein, und der Graph **zerfällt** in Teile (zwischen ihnen gibt es keinen Weg); zu groß, und Kanten **springen quer über die
Krümmung** (Kurzschlüsse) - dann verlaufen die Wege zunehmend in der Luftlinie und die Einbettung verliert an Qualität. Auch Rauschen erzeugt Kurzschlüsse, und die
Rechenzeit wächst kubisch mit der Zahl der Touren. Die Regler links zeigen jede dieser Schwächen live.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider("Anzahl Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=50)
    q = st.slider(
        "Wahre Anzahl versteckter Faktoren (q)", *bounds("q_slider"), key="q_slider",
        help="So viele echte Einflussgrößen erzeugen die 12 Kennzahlen. Mit mehr Faktoren wird die Fläche höherdimensional - bei gleich vielen Touren wird die Stichprobe dünner.",
    )
    curvature = st.slider(
        "Krümmung", *bounds("curvature_slider"), key="curvature_slider", step=0.05,
        help="0 = die Kennzahlen hängen linear von den Faktoren ab (dann hat Isomap keinen Vorteil vor der PCA). Größer = die Touren liegen auf einer zunehmend gebogenen Fläche.",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Messrauschen je Kennzahl. Rauschen lässt Nachbarn über die Fläche 'springen' und erzeugt Kurzschlüsse.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Nachbarschaftsgraph**")
    k = st.slider(
        "Nachbarn k", *bounds("k_slider"), key="k_slider",
        help="Jede Tour wird mit ihren k nächsten Nachbarn verbunden. Zu klein: der Graph zerfällt. Zu groß: Kurzschluss-Kanten quer über die Krümmung.",
    )

    st.button("🎲 Neue Touren generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren.")

sync_query_params(n_tours, q, curvature, noise, k, seed)

params = (int(n_tours), int(q), float(curvature), float(noise), int(seed))
with st.spinner("Berechne Nachbarschaftsgraph, kürzeste Wege und Einbettung..."):
    dataset = _dataset(*params)
    analysis = _analysis(*params, int(k))
res = analysis.result
z_color = dataset.z[:, 0]
data_key = params + (int(k),)

# --- Isomap in Aktion ---------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Isomap in Aktion")
st.caption(
    "Die 2-D-Ansicht zeigt die Touren in den ersten beiden Hauptkomponenten (Farbe = versteckter Faktor 1) - nur als Zeichenfläche; Isomap selbst rechnet in allen 12 Dimensionen. "
    "Wo die Fläche gebogen ist, liegen in dieser Ansicht Teile übereinander."
)
if "iso_step" not in st.session_state or st.session_state.get("iso_step_owner") != data_key:
    st.session_state["iso_step"] = 1
    st.session_state["iso_step_owner"] = data_key
step_col, play_col = st.columns([5, 1])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="iso_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

indices = res.indices
pos_pc1 = analysis.pca_2d[indices, 0]
source_local, target_local = int(np.argmin(pos_pc1)), int(np.argmax(pos_pc1))
weights_sub = res.weights[np.ix_(indices, indices)]
route_local = shortest_path_route(weights_sub, res.geodesic, source_local, target_local)
route = indices[route_local]
source, target = int(indices[source_local]), int(indices[target_local])
air = float(res.euclid[source, target])
path_len = float(res.geodesic[source_local, target_local])
rng = np.random.default_rng(0)
m = len(indices)
pair_a = rng.integers(0, m, size=min(1500, m * (m - 1) // 2))
pair_b = rng.integers(0, m, size=len(pair_a))
keep = pair_a != pair_b
pair_a, pair_b = pair_a[keep], pair_b[keep]

view_slot = st.empty()


def _render(current_step):
    if current_step == 1:
        view_slot.plotly_chart(build_graph_figure(analysis.pca_2d, res.edges, z_color), width="stretch", key=f"iso_view_{current_step}")
    elif current_step == 2:
        view_slot.plotly_chart(
            build_graph_figure(analysis.pca_2d, res.edges, z_color, route=route, straight=(source, target)), width="stretch",
            key=f"iso_view_{current_step}",
        )
    elif current_step == 3:
        view_slot.plotly_chart(
            build_distance_scatter(res.euclid[indices[pair_a], indices[pair_b]], res.geodesic[pair_a, pair_b], highlight=(air, path_len)),
            width="stretch", key=f"iso_view_{current_step}",
        )
    else:
        with view_slot.container():
            c1, c2 = st.columns(2)
            c1.markdown("**Isomap: Einbettung aus den geodätischen Abständen**")
            c1.plotly_chart(build_embedding(analysis.iso_2d, z_color[indices], "Isomap-Koordinate 1", "Isomap-Koordinate 2"), width="stretch", key="iso_embed_step")
            c2.markdown("**Zum Vergleich: PCA**")
            c2.plotly_chart(build_embedding(analysis.pca_2d, z_color, "PC1", "PC2"), width="stretch", key="pca_embed_step")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.0)
    step = 4
else:
    _render(step)

if step == 1:
    st.caption(f"Graph mit **k = {res.k}**: {len(res.edges):,} Kanten zwischen {res.n} Touren. Ist er nicht zusammenhängend, zerfällt er in Teile - dann gibt es zwischen ihnen keinen Weg.".replace(",", "."))
elif step == 2:
    st.caption(
        f"Zwischen den beiden weitest entfernten Touren (Sterne) liegt die **Luftlinie** bei {air:.1f}, der **kürzeste Weg** durch den Graphen bei {path_len:.1f} "
        f"({path_len / air:.2f}× so lang). Je gebogener die Fläche, desto größer der Umweg."
    )
elif step == 3:
    st.caption(
        "Jeder Punkt ein Tourenpaar. Bei einer geraden Fläche liegen alle Punkte auf der Diagonalen (Weg = Luftlinie). Über der Diagonalen ist die Fläche gekrümmt - "
        "genau diesen Umweg misst Isomap, die PCA nicht."
    )
else:
    st.caption("Farbe = versteckter Faktor 1. Verläuft sie in der Isomap-Einbettung glatt und ohne Überlappung, hat Isomap die Fläche entrollt.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was Isomap gefunden hat")
if not res.connected:
    st.warning(
        f"⚠️ Der Graph ist mit k = {res.k} **nicht zusammenhängend**: {int(res.components.max()) + 1} Teile. Isomap bettet nur den größten Teil ein "
        f"({len(indices)} von {res.n} Touren) - die übrigen {res.n_dropped} fehlen im Ergebnis, und alle Kennzahlen unten beziehen sich nur auf die eingebetteten Touren."
    )
e1, e2, e3, e4 = st.columns(4)
e1.metric("Eingebettete Touren", f"{len(indices)} / {res.n}")
e2.metric("Dimensionsschätzung", "–" if analysis.q_hat is None else analysis.q_hat, help=f"Isomaps Schätzung der Dimension (Knick der Residualvarianz): kleinste Dimension mit Residualvarianz ≤ {RV_TARGET:g}. Wahre Dimension: q = {dataset.q}.")
e3.metric("R² der wahren Faktoren", f"{analysis.r2_iso:.2f}", delta=f"{analysis.r2_iso - analysis.r2_pca:+.2f} ggü. PCA", delta_color="normal",
          help="Wie gut lassen sich die versteckten Faktoren aus den ersten zwei Koordinaten zurückgewinnen (quadratische Regression). PCA mit derselben Messung im Delta.")
e4.metric("Trustworthiness", f"{analysis.trust_iso:.2f}", delta=f"{analysis.trust_iso - analysis.trust_pca:+.2f} ggü. PCA", delta_color="normal",
          help=f"Nachbarschaft erhalten: Anteil der Nachbarn in der 2-D-Einbettung, die auch im Originalraum Nachbarn sind (k = {C.TRUST_NEIGHBORS}); 1 = perfekt.")
rc1, rc2 = st.columns([3, 2])
with rc1:
    st.markdown("**Residualvarianz: wie viele Dimensionen braucht Isomap?**")
    st.plotly_chart(build_residual_curve(analysis.rv, dataset.q, RV_TARGET), width="stretch", key="residual_curve")
with rc2:
    st.markdown("**Einbettung (Isomap)**")
    st.plotly_chart(build_embedding(analysis.iso_2d, z_color[indices], "Isomap-Koordinate 1", "Isomap-Koordinate 2"), width="stretch", key="iso_embedding")
st.caption(
    "Residualvarianz = 1 − R² zwischen den geodätischen Abständen und den Abständen der ersten d Koordinaten: sie fällt, bis alle wesentlichen Dimensionen enthalten sind - "
    "der Knick ist Isomaps Schätzung der wahren Dimension."
)

st.markdown("---")

# --- 📐 Wie stark hängt das Ergebnis von k ab? ----------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von k ab?")
st.markdown(
    """
*k* ist Isomaps einziger wichtiger Regler - und er hat ein **Fenster**: zu klein, und der Graph zerfällt; zu groß, und Kurzschluss-Kanten überbrücken die
Krümmung. Live für Ihr aktuelles Szenario über **feste Sweep-Seeds** (unabhängig vom Demo-Seed) geprüft, nicht behauptet:
"""
)
level, code, vd = verdict(analysis, dataset, int(k))
if code == "disconnected":
    st.warning(
        f"⚠️ **Der Graph zerfällt** (k = {vd['k']}): {vd['components']} Teile, {vd['dropped']} Touren fehlen in der Einbettung. Ein größeres k verbindet die Teile - "
        "die Kurve unten zeigt, ab wann der Graph zusammenhängt."
    )
elif code == "short_circuit":
    st.warning(
        f"⚠️ **Kurzschlüsse**: bei k = {vd['k']} liegt die Residualvarianz bei {vd['q']} Dimensionen bei {vd['rv_q']:.2f} (gut wäre unter 0.1), {vd['short'] * 100:.1f} % der Kanten überspringen die Krümmung. "
        f"Die Wege verlaufen zunehmend in der Luftlinie - R² der Faktoren nur noch {vd['r2_iso']:.2f} (bei gut gewähltem k liegt es deutlich höher; PCA: {vd['r2_pca']:.2f})."
    )
elif code == "noise_short_circuit":
    st.warning(
        f"⚠️ **Rauschen erzeugt Kurzschlüsse**: schon bei k = {vd['k']} überspringen {vd['short'] * 100:.1f} % der Kanten die Krümmung, die Residualvarianz bei {vd['q']} Dimensionen liegt bei "
        f"{vd['rv_q']:.2f}. Rauschen lässt Nachbarn im Merkmalsraum über die Fläche 'springen' - ein kleineres k oder weniger Rauschen hilft."
    )
elif code == "no_advantage":
    st.info(
        f"ℹ️ **Kein Vorteil vor der PCA**: die Daten sind gerade (Krümmung 0) - R² der Faktoren {vd['r2_iso']:.2f} (Isomap) gegen {vd['r2_pca']:.2f} (PCA). Isomap liefert dasselbe "
        "und kostet ein Vielfaches an Rechenzeit."
    )
elif code == "isomap_wins":
    st.success(
        f"✅ **Isomap entrollt die Fläche**: R² der Faktoren {vd['r2_iso']:.2f} gegen {vd['r2_pca']:.2f} bei der PCA, Trustworthiness {vd['trust_iso']:.2f} gegen {vd['trust_pca']:.2f}, "
        f"Residualvarianz bei {vd['q']} Dimensionen nur {vd['rv_q']:.3f}."
    )
else:
    st.info(
        f"Isomap erreicht R² {vd['r2_iso']:.2f} (PCA {vd['r2_pca']:.2f}), Residualvarianz bei {vd['q']} Dimensionen {vd['rv_q']:.2f} - kein klarer Gewinn und kein klarer Bruch."
    )

sweep_rows = _sweep(int(q), float(curvature), float(noise))
st.plotly_chart(build_k_sweep(sweep_rows, int(k)), width="stretch", key="k_sweep")
good = [r["k"] for r in sweep_rows if r["connected"] >= 1.0 and r["rv_q"] <= 0.05]
st.caption(
    f"Gleiche Einstellungen (q = {dataset.q}, Krümmung {curvature:.2f}, Rauschen {noise:.2f}), nur k wächst; Mittel über {len(C.SWEEP_SEEDS)} feste Seeds mit je {C.SWEEP_N_TOURS} Touren. "
    + (
        f"Zusammenhängend **und** Residualvarianz bei q höchstens 0.05: k = {min(good)} … {max(good)} (aus den getesteten Werten {', '.join(str(r['k']) for r in sweep_rows)})."
        if good else "Für diese Einstellungen erreicht kein getestetes k Residualvarianz ≤ 0.05 bei zusammenhängendem Graphen."
    )
)

st.markdown("---")

# --- Rechenzeit ------------------------------------------------------------------------------------------------------------

st.markdown("## ⏱️ Rechenzeit: die dritte Schwäche")
st.caption(
    "Kürzeste Wege zwischen allen Paaren (Floyd-Warshall) und eine Eigenzerlegung einer n×n-Matrix kosten beide **kubische** Zeit - die PCA zerlegt nur eine 12×12-Matrix. "
    "Außerdem kennt Isomap keine Abbildung für **neue** Touren: ein zusätzlicher Punkt braucht eine neue Berechnung (oder eine Zusatzmethode, die hier nicht gezeigt wird)."
)
if "timing_rows" not in st.session_state:
    if st.button("⏱️ Rechenzeit messen (ca. 5 s)", key="timing_start", help=f"Misst Isomap (k = {C.TIMING_K}) und PCA für n = {', '.join(str(n) for n in C.TIMING_NS)} auf diesem Rechner."):
        with st.spinner("Messe..."):
            st.session_state["timing_rows"] = timing_sweep()
        st.rerun()
else:
    rows = st.session_state["timing_rows"]
    st.plotly_chart(build_timing(rows), width="stretch", key="timing_chart")
    ns = np.array([r["n"] for r in rows], dtype=float)
    iso = np.array([r["isomap"] for r in rows])
    exponent = float(np.polyfit(np.log(ns[-3:]), np.log(iso[-3:]), 1)[0])
    st.table({
        "Touren n": [r["n"] for r in rows],
        "Isomap": [f"{r['isomap']:.3f} s" for r in rows],
        "PCA": [f"{r['pca'] * 1000:.2f} ms" for r in rows],
        "Faktor": [f"{r['isomap'] / max(r['pca'], 1e-9):,.0f}×".replace(",", ".") for r in rows],
    })
    st.caption(
        f"Gemessen auf diesem Rechner (Wandzeit, k = {C.TIMING_K}, ein Lauf je n): über die letzten drei Punkte wächst die Isomap-Zeit etwa mit n^{exponent:.1f}. Die genaue Steigung "
        "hängt von Rechner und Zwischenspeichern ab."
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Geodätische Abstände.** Gegeben $n$ Punkte $x_i \in \mathbb{R}^d$ (hier: z-Werte der 12 Kennzahlen). Der Nachbarschaftsgraph $G$ verbindet $x_i$ und $x_j$, wenn einer unter den
$k$ nächsten Nachbarn des anderen ist; das Kantengewicht ist $\lVert x_i - x_j \rVert$. Der geodätische Abstand wird durch die Länge des kürzesten Weges in $G$ genähert:

$$
d_G(i,j) = \min_{\text{Wege } i = p_0, p_1, \dots, p_m = j} \ \sum_{l=1}^{m} \lVert x_{p_{l-1}} - x_{p_l} \rVert .
$$

Für dichte Stichproben einer glatten, isometrisch entrollbaren Fläche konvergiert $d_G$ gegen den wahren geodätischen Abstand auf der Fläche (Bernstein, de Silva, Langford & Tenenbaum, 2000).
Die Demo berechnet $d_G$ mit Floyd-Warshall: $d^{(m)}_{ij} = \min(d^{(m-1)}_{ij},\ d^{(m-1)}_{im} + d^{(m-1)}_{mj})$, Aufwand $O(n^3)$.

**Klassisches MDS.** Aus $D = (d_G(i,j))$ folgt die doppelt zentrierte Matrix $B = -\tfrac12\, J D^{\circ 2} J$ mit $J = I - \tfrac1n \mathbf{1}\mathbf{1}^\top$ ($D^{\circ 2}$ = elementweise Quadrate). Mit der
Eigenzerlegung $B = V \Lambda V^\top$ sind die Koordinaten der ersten $m$ Dimensionen $Y = V_m \Lambda_m^{1/2}$. Wären die Abstände exakt euklidisch, reproduziert $Y$ sie exakt (siehe Test).

**Residualvarianz.** $1 - R^2\big(d_G,\ d_{Y_{1..m}}\big)$, wobei $R$ der Korrelationskoeffizient zwischen den geodätischen Abständen und den euklidischen Abständen der ersten $m$ Koordinaten ist
(Tenenbaum et al., 2000). Der Knick in $m$ schätzt die Dimension der Fläche.

**Grenzen.** (1) *Kurzschlüsse*: eine Kante, die zwei im Faktorraum weit entfernte Touren verbindet, verkürzt viele Wege und verzerrt die ganze Einbettung - bei zu großem $k$ oder viel Rauschen.
(2) *Zusammenhang*: für zu kleines $k$ zerfällt $G$; zwischen den Teilen ist $d_G = \infty$. Die Demo bettet dann nur die größte Komponente ein und meldet den Rest.
(3) *Konvexität*: Isomap erwartet eine Fläche, die sich isometrisch in einen konvexen Bereich entrollen lässt; Löcher und nicht isometrische Krümmung verzerren das Ergebnis.
(4) *Rechenzeit* $O(n^3)$ und *Speicher* $O(n^2)$. (5) *Kein Out-of-sample*: neue Punkte haben keine Koordinaten, ohne die Einbettung neu zu berechnen.

**Trustworthiness** (Venna & Kaski, 2001): $T = 1 - \frac{2}{nk(2n - 3k - 1)} \sum_i \sum_{j \in U_i} (r(i,j) - k)$ mit $U_i$ = Nachbarn in der Einbettung, die im Originalraum keine sind, und $r(i,j)$ ihrem Originalrang.

**Kurzschluss-Anteil** (nur messbar, weil die wahren Faktoren $z$ bekannt sind): Anteil der Graphkanten, deren Länge im Faktorraum mehr als das Dreifache dessen beträgt, was ihre Länge im Merkmalsraum bei lokalem
Maßstab erwarten lässt (Maßstab = Median von $\lVert z_i - z_j \rVert / \lVert x_i - x_j \rVert$ über die Kanten des 3-NN-Graphen).

Implementiert in `iso_algorithm.py` (Graph, kürzeste Wege, MDS, Residualvarianz), `iso_scenario.py` (Lieferrouten-Generator, wortgleich aus pca-demo) und `iso_evaluation.py` (Kennzahlen, Sweep, Verdict, Zeitmessung).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Dimensionsreduktion: von PCA bis Autoencoder](https://sebastianhanisch.net/konzepte-dimensionsreduktion.html)."
)
