"""Defaults, Slider-Grenzen und Presets für die Isomap-Demo. Merkmale und Erzeugungs-Konstanten sind wortgleich aus pca-demo übernommen
(dieselben Lieferrouten - dieselbe gekrümmte Fläche, an der PCA scheiterte); alles Übrige ist neu."""

# --- Merkmale: 12 Kennzahlen je Tour in 4 Gruppen zu je 3 (Name, Einheit, Mittelwert, typische Streuung in Einheiten) ------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_FEATURES = len(FEATURES)
FEATURE_NAMES = tuple(f[0] for f in FEATURES)
FEATURE_LABELS = tuple(f"{f[0]} [{f[1]}]" for f in FEATURES)
GROUPS = ("Größe", "Zeitdruck", "Verkehr", "Sonderfälle")     # je 3 aufeinanderfolgende Merkmale
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_FEATURES))

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 100, 600
DEFAULT_Q = 2
Q_MIN, Q_MAX = 1, 4
DEFAULT_CURVATURE = 1.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_K = 10
K_MIN, K_MAX = 1, 60
DEFAULT_SEED = 7

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
OUTLIER_SCALE = 10.0                   # Sonderfahrten: latenter Faktor um diesen Faktor vergrößert
CROSS_LOADING = 0.15                   # kleine Querladungen zwischen Merkmalsgruppen
WITHIN_LOADINGS = (0.95, 0.9, 0.85)    # Ladung der drei Merkmale einer Gruppe auf ihren Faktor
CURVATURE_FREQUENCY = 1.6              # Frequenz der sin/cos-Terme der Krümmung
CURVATURE_AMPLITUDE = 2.0              # Länge jeder Spalte der Krümmungsmatrix (in z-Einheiten bei Krümmung 1)
LAYOUT_SEED = 20240915                 # feste Ladungs- und Krümmungsmatrizen (unabhängig vom Seed der Touren)


# --- Auswertung --------------------------------------------------------------------------------------------------------
N_COMPONENTS_MAX = 10                  # Residualvarianz-Kurve bis zu dieser Dimension
TRUST_NEIGHBORS = 10
SWEEP_SEEDS = tuple(100_000 + i for i in range(3))                 # feste Sweep-Seeds, unabhängig vom Demo-Seed
SWEEP_KS = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 45)
SWEEP_N_TOURS = 250
TIMING_NS = (100, 200, 400, 600, 800, 1000)
TIMING_K = 10

_BASE = {"n_tours": DEFAULT_N_TOURS, "q": 2, "curvature": 1.0, "noise": DEFAULT_NOISE, "k": 8, "seed": DEFAULT_SEED}
PRESETS = {
    "Gekrümmte Fläche: Isomap gewinnt": {**_BASE},
    "Gerade Daten: kein Vorteil": {**_BASE, "curvature": 0.0, "k": 10},
    "Zu kleines k: Graph zerfällt": {**_BASE, "k": 2},
    "Zu großes k: Kurzschlüsse": {**_BASE, "k": 50},
    "Rauschen erzeugt Kurzschlüsse": {**_BASE, "noise": 0.8, "k": 10},
    "Drei Faktoren: dünne Stichprobe": {**_BASE, "q": 3, "k": 10},
}
PRESET_HELP = {
    "Gekrümmte Fläche: Isomap gewinnt": "Dieselbe gebogene Fläche wie in der PCA-Demo: die Faktoren lassen sich aus Isomaps Einbettung fast vollständig zurückgewinnen (R² ≈ 0.98), aus der PCA nur zur Hälfte (≈ 0.50) - die Residualvarianz hat einen scharfen Knick bei zwei Dimensionen.",
    "Gerade Daten: kein Vorteil": "Ohne Krümmung sind Weg und Luftlinie gleich lang: Isomap liefert dasselbe wie die PCA (R² ≈ 0.97 gegen 0.98) - und kostet ein Vielfaches an Rechenzeit.",
    "Zu kleines k: Graph zerfällt": "Mit nur zwei Nachbarn je Tour reißt der Graph in mehrere Teile: zwischen ihnen gibt es keinen Weg, Isomap bettet nur den größten Teil ein und meldet, wie viele Touren fehlen.",
    "Zu großes k: Kurzschlüsse": "Mit 50 Nachbarn springen Kanten quer über die Krümmung: die Wege verlaufen fast in der Luftlinie, die Residualvarianz bei zwei Dimensionen steigt über 0.1 und die Einbettung verliert an Qualität (R² ≈ 0.85 statt 0.98).",
    "Rauschen erzeugt Kurzschlüsse": "Mit viel Rauschen liegen Nachbarn im Merkmalsraum nicht mehr auf derselben Stelle der Fläche: schon bei k = 10 überspringen Kanten die Krümmung und die Residualvarianz steigt.",
    "Drei Faktoren: dünne Stichprobe": "Bei drei versteckten Faktoren ist die Fläche höherdimensional, 300 Touren tasten sie nur dünn ab: Kurzschlüsse treten schon bei mittlerem k auf, Isomap ist zwar besser als die PCA, entrollt die Fläche aber nicht sauber.",
}
# Erwartete Messwerte je Preset (mit dem ausgelieferten Code kalibriert; tests/test_presets.py prüft sie). Tupel = (min, max), Text = Verdict-Code.
PRESET_EXPECTED_BANDS = {
    "Gekrümmte Fläche: Isomap gewinnt": {"verdict": "isomap_wins", "r2_iso": (0.95, 1.0), "r2_pca": (0.42, 0.58), "rv_q": (0.0, 0.05), "q_hat": (2, 2), "trust_iso": (0.98, 1.0)},
    "Gerade Daten: kein Vorteil": {"verdict": "no_advantage", "r2_iso": (0.94, 1.0), "r2_pca": (0.95, 1.0)},
    "Zu kleines k: Graph zerfällt": {"verdict": "disconnected", "dropped": (50, 100), "components": (5, 12)},
    "Zu großes k: Kurzschlüsse": {"verdict": "short_circuit", "rv_q": (0.11, 0.25), "r2_iso": (0.75, 0.92)},
    "Rauschen erzeugt Kurzschlüsse": {"verdict": "noise_short_circuit", "rv_q": (0.11, 0.25), "short": (0.008, 0.05)},
    "Drei Faktoren: dünne Stichprobe": {"verdict": "short_circuit", "r2_iso": (0.40, 0.62), "r2_pca": (0.10, 0.28)},
}
