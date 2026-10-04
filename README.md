# Isomap an Lieferrouten-Kennzahlen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-isomap-demo.streamlit.app/)**

Zweites Stück der **Dimensionsreduktion-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Isomap** – an einem wachsenden
Beispiel. Vehikel: **dieselben 12 Lieferrouten-Kennzahlen wie in [pca-demo](../pca-demo)** (der Generator ist wortgleich kopiert und per Test gegen dessen Ausgabe eingefroren),
erzeugt aus wenigen versteckten Faktoren – dieselbe gekrümmte Fläche, an der PCA scheiterte.

**Einordnung in die Reihe (die Kanten des Graphen):** Isomap **behebt die Linearitätsschwäche der PCA** (pca-demo): statt gerader Abstände verwendet es
**geodätische** Abstände entlang eines Nachbarschaftsgraphen und entrollt so die gebogene Fläche. Es hat dafür **eigene Schwächen**, die die Demo live zeigt:
```
pca-demo → isomap-demo   (Linearität → geodätische Abstände; eigene Schwäche: Kurzschluss-Kanten, Zusammenhang, Rauschen, O(n³), kein Out-of-sample)
pca-demo → LLE | t-SNE → UMAP → PaCMAP | Autoencoder   (weitere Äste, inzwischen gebaut: lle-demo, tsne-demo, umap-demo, pacmap-demo, autoencoder-demo)
```

## Was die Demo zeigt

1. **Isomap in Aktion** (Schritt-Slider + Abspielen): Nachbarschaftsgraph → kürzester Weg zwischen den beiden am weitesten entfernten Touren (gegen die Luftlinie) →
   Luftlinie gegen Weg für viele Tourenpaare → Einbettung neben der PCA.
2. **Was Isomap gefunden hat:** eingebettete Touren, Dimensionsschätzung (Knick der **Residualvarianz**), R² der wahren Faktoren gegen PCA, Trustworthiness gegen PCA.
3. **📐 Wie stark hängt das Ergebnis von k ab?** (live über feste Sweep-Seeds ab 100000, unabhängig vom Demo-Seed): Residualvarianz bei q Dimensionen, R², Kurzschluss-Anteil und
   Zusammenhang je k, mit Verdict (Graph zerfällt → Kurzschlüsse / Rauschen → kein Vorteil → Isomap gewinnt).
4. **⏱️ Rechenzeit** (Knopf, gemessen): Isomap gegen PCA für wachsende n.

Messwerte (Seed 7, 300 Touren, q = 2; die Presets prüfen sie mit Bändern):

| Situation | Messung |
|---|---|
| Gekrümmte Fläche, k = 8 | R² der wahren Faktoren **0.98** gegen 0.50 (PCA); Trustworthiness 1.00 gegen 0.86; Residualvarianz bei 2 Dimensionen 0.016, Knick bei 2 |
| Gerade Daten (Krümmung 0), k = 10 | **kein Vorteil**: R² 0.97 gegen 0.98 (PCA), und ein Vielfaches an Rechenzeit |
| k = 2 | **Graph zerfällt** in 8 Teile, 73 von 300 Touren fehlen in der Einbettung |
| k = 50 | **Kurzschlüsse**: Residualvarianz bei 2 Dimensionen 0.16, R² fällt auf 0.85 |
| Rauschen 0.8, k = 10 | 2.0 % der Kanten überspringen die Krümmung, Residualvarianz 0.16, R² 0.84 |
| q = 3 Faktoren, k = 10 | dünne Stichprobe: R² 0.51 gegen 0.19 (PCA), aber Kurzschlüsse, Residualvarianz 0.12 |

**Das Fenster für k** (gekrümmte Daten, 3 feste Seeds × 250 Touren): Residualvarianz bei q Dimensionen ≈ 0.01–0.02 für k = 6…10, 0.075 bei k = 15, 0.13 bei k = 20 und 0.25 bei k = 60;
R² der Faktoren fällt von 0.98 auf 0.79, der Kurzschluss-Anteil steigt von etwa 0.04 % (k = 10) auf 1.5 % (k = 60). Auf geraden Daten schadet ein großes k dagegen nicht (Residualvarianz fällt weiter, kein Kurzschluss) -
die Kurzschlüsse entstehen erst durch die Krümmung.

**Rechenzeit** (lokale Messung, k = 10, ein Lauf je n): 100 Touren ≈ 3 ms, 600 Touren ≈ 0.4 s, 1000 Touren ≈ 1.6-2 s bei PCA im Bereich 0.1-0.6 ms; die Steigung über die letzten
drei Punkte liegt bei etwa n^2.7 (theoretisch n³: Floyd-Warshall und Eigenzerlegung; numpy-Vektorisierung und Speicher drücken die gemessene Steigung).

## Modell und Verfahren

- **Generator** (`iso_scenario.py`): wortgleich aus pca-demo; latente Faktoren, 12 Merkmale in 4 Gruppen, Krümmung `κ·B·h(z)`, Rauschen. Isomap arbeitet immer auf z-Werten (die
  Standardisierung der PCA-Demo ist hier fest), Sonderfahrten und Rohdaten-Skalierung entfallen.
- **Isomap** (`iso_algorithm.py`, numpy, ohne sklearn): symmetrischer kNN-Graph → Floyd-Warshall (vektorisiert über den Zwischenknoten, O(n³)) → klassisches MDS (Doppelzentrierung, `eigh`).
  Bei nicht zusammenhängendem Graph wird nur die **größte Komponente** eingebettet und der Rest gemeldet - nichts wird still repariert.
- **Auswertung** (`iso_evaluation.py`): Residualvarianz (Tenenbaum et al. 2000); **R² der wahren Faktoren** aus den ersten zwei Koordinaten per quadratischer Regression (damit eine
  monotone Umparametrisierung nicht bestraft wird), PCA mit derselben Messung; Trustworthiness (Venna & Kaski, eigene Implementierung); **Kurzschluss-Anteil** = Anteil der Kanten, deren
  Länge im Faktorraum mehr als das Dreifache dessen beträgt, was ihre Länge im Merkmalsraum bei lokalem Maßstab erwarten lässt (nur messbar, weil die wahren Faktoren bekannt sind).

## Was nicht funktioniert hat

- **Aufgerollte Fläche (Swiss Roll) als zusätzlicher Regler:** ausprobiert und wieder verworfen. Mit 300-400 Touren entrollt Isomap die Spirale nur in einem schmalen, seedabhängigen k-Fenster
  (Anteil der Seeds mit Residualvarianz < 0.1 bei 400 Touren, Rauschen 0.1: 4/5 bei k = 5, aber nur 1/5 bei k = 20 und 0/5 bei k = 45); die Demo hätte auf wenige Presets zugeschnitten werden müssen. Die
  glatte Krümmung aus pca-demo zeigt dieselben Effekte (Fenster für k, Kurzschlüsse) robust über Seeds.
- **Kurzschluss-Anteil nach absoluter latenter Kantenlänge:** liefert für gerade und gekrümmte Daten dieselben Werte (er misst nur die Kantenlänge) - deshalb der lokale Maßstab.

## Grenzen (Text, nicht gemessen)

- **Kein Out-of-sample:** ein neuer Punkt hat keine Koordinaten, ohne die Einbettung neu zu berechnen.
- **Konvexität/Löcher:** Isomap erwartet eine isometrisch entrollbare, konvexe Fläche; Löcher und nicht isometrische Krümmung verzerren das Ergebnis.

## Verifikation

- Isomap gegen `sklearn.manifold.Isomap`: geodätische Distanzmatrix identisch (1e-8), Einbettung bis auf Drehung/Spiegelung gleich; Floyd-Warshall gegen `scipy`-Dijkstra und Handgraph.
- Geodäten symmetrisch, Dreiecksungleichung, nie kürzer als die Luftlinie; kNN-Graph symmetrisch; Komponenten-Erkennung (Handgraph); klassisches MDS reproduziert euklidische Abstände exakt.
- Generator bit-identisch zu pca-demo (eingefrorene Referenzwerte); Trustworthiness gegen `sklearn.manifold.trustworthiness` (1e-9).
- Sweep über feste Seeds deterministisch; auf gekrümmten Daten Fenster für k, auf geraden Daten kein Schaden durch großes k; Verdict-Codes; alle 6 Presets in kalibrierten Bändern; AppTest-Rauchtests
  (Default, jedes Preset, jeder Schritt, Randgrößen, Schritt-Zustand), Achsensperre aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 k-Sweep, Rechenzeit, Mathe |
| `iso_algorithm.py` | Isomap von Grund auf (Graph, kürzeste Wege, MDS, Residualvarianz) |
| `iso_scenario.py`, `iso_constants.py` | Lieferrouten-Generator (wortgleich aus pca-demo), Konstanten, Presets |
| `iso_evaluation.py` | Kennzahlen, Sweep, Verdict, Zeitmessung |
| `iso_presets.py`, `iso_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | sklearn-/scipy-Kreuzvergleiche, Generator-Referenz, Auswertung, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Dimensionsreduktion: von PCA bis Autoencoder](https://sebastianhanisch.net/konzepte-dimensionsreduktion.html).
