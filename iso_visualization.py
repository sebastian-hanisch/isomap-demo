"""Plotly-Visualisierungen der Isomap-Demo: Nachbarschaftsgraph mit kürzestem Weg, Luftlinie gegen Weg, Einbettung neben PCA, Residualvarianz,
k-Sweep und Rechenzeit. Alle Figuren laufen durch `lock_axes` (Touch-Scrolling-Konvention des Portfolios: keine Zoom-/Pan-Gesten im Chart)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, GREEN, RED, GRAY = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _scatter(coords, color, name="Touren", size=7, showscale=False, label="latenter Faktor 1"):
    return go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers", name=name, hoverinfo="skip",
        marker=dict(color=color, colorscale="Viridis", size=size, showscale=showscale, line=dict(width=0.5, color="white"),
                    colorbar=dict(title=label) if showscale else None),
    )


def build_graph_figure(coords, edges, color, route=None, straight=None, show_edges=True):
    """Datenpunkte in der 2-D-Ansicht (erste zwei Hauptkomponenten) mit den Kanten des Nachbarschaftsgraphen; optional der kürzeste Weg (orange)
    und die Luftlinie (grau gestrichelt) zwischen zwei Touren."""
    fig = go.Figure()
    if show_edges and len(edges):
        xs, ys = [], []
        for i, j in edges:
            xs += [coords[i, 0], coords[j, 0], None]
            ys += [coords[i, 1], coords[j, 1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(120,120,120,0.35)", width=1), name="Graphkanten", hoverinfo="skip"))
    fig.add_trace(_scatter(coords, color, showscale=True))
    if straight is not None:
        a, b = straight
        fig.add_trace(go.Scatter(x=[coords[a, 0], coords[b, 0]], y=[coords[a, 1], coords[b, 1]], mode="lines", name="Luftlinie",
                                 line=dict(color=GRAY, width=3, dash="dash"), hoverinfo="skip"))
    if route is not None:
        fig.add_trace(go.Scatter(x=coords[route, 0], y=coords[route, 1], mode="lines+markers", name="kürzester Weg im Graphen",
                                 line=dict(color=ORANGE, width=4), marker=dict(size=8, color=ORANGE), hoverinfo="skip"))
    if straight is not None:
        pts = np.array(straight)
        fig.add_trace(go.Scatter(x=coords[pts, 0], y=coords[pts, 1], mode="markers", name="Start / Ziel", hoverinfo="skip",
                                 marker=dict(color=RED, size=13, symbol="star", line=dict(width=1, color="#14233B"))))
    fig.update_xaxes(title="PC1 (Ansicht)")
    fig.update_yaxes(title="PC2 (Ansicht)")
    fig.update_layout(template="plotly_white", height=460, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_distance_scatter(euclid_pairs, geodesic_pairs, highlight=None):
    """Luftlinie (euklidisch) gegen Weg im Graphen (geodätisch) für zufällige Touren-Paare: auf der Diagonalen ist der Umweg 1, darüber ist die Fläche gekrümmt."""
    top = float(max(euclid_pairs.max(), geodesic_pairs.max())) * 1.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=euclid_pairs, y=geodesic_pairs, mode="markers", name="Touren-Paare", hoverinfo="skip",
                             marker=dict(color=BLUE, size=5, opacity=0.4)))
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Weg = Luftlinie", line=dict(color=GRAY, dash="dash"), hoverinfo="skip"))
    if highlight is not None:
        fig.add_trace(go.Scatter(x=[highlight[0]], y=[highlight[1]], mode="markers", name="Start-Ziel-Paar", hoverinfo="skip",
                                 marker=dict(color=RED, size=14, symbol="star", line=dict(width=1, color="#14233B"))))
    fig.update_xaxes(title="Luftlinie (euklidischer Abstand, z-Werte)", range=[0, top])
    fig.update_yaxes(title="Weg im Graphen (geodätischer Abstand)", range=[0, top])
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_embedding(coords, color, title_x, title_y, label="latenter Faktor 1"):
    fig = go.Figure(_scatter(coords, color, showscale=True, label=label, size=7))
    fig.update_xaxes(title=title_x)
    fig.update_yaxes(title=title_y)
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=20, b=10))
    return lock_axes(fig)


def build_residual_curve(rv, q, target):
    """Residualvarianz gegen die Anzahl Dimensionen: der Knick zeigt Isomaps Schätzung der Dimension; die wahre Dimension q ist markiert."""
    dims = np.arange(1, len(rv) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dims, y=rv, mode="lines+markers", line=dict(color=ORANGE, width=3), name="Residualvarianz",
                             hovertemplate="%{x} Dim.: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=target, line_dash="dot", line_color=GRAY, annotation_text=f"Knick-Schwelle {target:g}", annotation_position="top right")
    fig.add_vline(x=q, line_dash="dash", line_color=GREEN, annotation_text=f"wahre Dimension q = {q}", annotation_position="top right")
    fig.update_xaxes(title="Anzahl Dimensionen der Einbettung", dtick=1)
    fig.update_yaxes(title="Residualvarianz (1 − R²)", rangemode="tozero")
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return lock_axes(fig)


def build_k_sweep(rows, current_k):
    """Sweep über die Nachbarzahl k (feste Seeds): Residualvarianz bei q, R² der wahren Faktoren, Kurzschluss-Anteil; das aktuelle k ist markiert."""
    ks = [r["k"] for r in rows]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Residualvarianz bei q Dimensionen", "R² der wahren Faktoren", "Kurzschluss-Anteil der Kanten"))
    fig.add_trace(go.Scatter(x=ks, y=[r["rv_q"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name="Residualvarianz"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ks, y=[r["r2"] for r in rows], mode="lines+markers", line=dict(color=BLUE, width=3), name="R²"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ks, y=[r["connected"] for r in rows], mode="lines+markers", line=dict(color=GRAY, width=2, dash="dot"),
                             name="Anteil zusammenhängender Graphen"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ks, y=[r["short"] * 100 for r in rows], mode="lines+markers", line=dict(color=RED, width=3), name="Kurzschluss-Anteil"), row=1, col=3)
    for col in (1, 2, 3):
        fig.add_vline(x=current_k, line_dash="dot", line_color=GREEN, row=1, col=col)
    fig.update_xaxes(title_text="Nachbarn k", type="log", tickvals=[2, 3, 5, 10, 20, 45], ticktext=["2", "3", "5", "10", "20", "45"])
    fig.update_yaxes(rangemode="tozero", row=1, col=1)
    fig.update_yaxes(range=[0, 1.02], row=1, col=2)
    fig.update_yaxes(title_text="% der Kanten", rangemode="tozero", row=1, col=3)
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_timing(rows):
    """Gemessene Rechenzeit gegen n (log-log): Isomap wächst etwa kubisch, PCA kaum. Referenzlinie ~ n³, durch den letzten Isomap-Punkt gelegt."""
    ns = np.array([r["n"] for r in rows], dtype=float)
    iso = np.array([r["isomap"] for r in rows])
    pca = np.array([max(r["pca"], 1e-6) for r in rows])
    ref = iso[-1] * (ns / ns[-1]) ** 3
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=iso, mode="lines+markers", name="Isomap", line=dict(color=ORANGE, width=3)))
    fig.add_trace(go.Scatter(x=ns, y=pca, mode="lines+markers", name="PCA", line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=ns, y=ref, mode="lines", name="∝ n³ (Referenz)", line=dict(color=GRAY, dash="dash")))
    fig.update_xaxes(title="Anzahl Touren n", type="log")
    fig.update_yaxes(title="Rechenzeit (s)", type="log")
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)
