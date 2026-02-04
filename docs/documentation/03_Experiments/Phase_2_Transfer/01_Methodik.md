# Phase 3.4: Transfer-Evaluation

## Einleitung

Die Transfer-Evaluation ist das **Herzstück der Forschungsfrage**: Wie gut generalisieren Modelle, die auf Berliner Bäumen trainiert wurden, auf Leipziger Bäume — ohne jegliche Anpassung?

Dieses Szenario ist praktisch hochrelevant: Eine Stadt mit umfassendem Baumkataster und Ressourcen für ML entwickelt Klassifikationsmodelle, die dann von anderen Städten "out-of-the-box" genutzt werden könnten.

---

## Forschungsfragen

1. **Transfer-Gap:** Wie viel Performance verlieren Berlin-optimierte Modelle bei Zero-Shot Transfer nach Leipzig?
2. **ML vs. NN:** Transferieren ML-Modelle oder Neuronale Netze besser?
3. **Per-Genus Robustheit:** Welche Baumgattungen transferieren gut, welche schlecht?

---

## Methodische Begründungen

### Warum Zero-Shot Transfer?

**Definition:** Zero-Shot Transfer bedeutet, dass das Modell auf Leipzig angewendet wird, ohne jemals Leipziger Daten gesehen zu haben.

```
Training:       Berlin Train + Berlin Val
Evaluation:     Leipzig Test

Leipzig-Daten während Training: KEINE
```

**Begründung:**

- Realistische Simulation: Neue Stadt hat kein Trainingsdaten-Budget
- Klare Baseline für Fine-Tuning: "Was bringt Fine-Tuning gegenüber Zero-Shot?"
- Wissenschaftliche Reinheit: Transfer-Gap ist eindeutig messbar

### Warum beide Champions testen?

```
ML Champion ──────┐
                  ├──→ Leipzig Test ──→ Vergleich ──→ Best Transfer Model
NN Champion ──────┘

Mögliche Outcomes:
├── ML transferiert besser: Weniger Overfitting, robustere Features
├── NN transferiert besser: Abstraktere Repräsentationen
└── Kein signifikanter Unterschied: Beide für Fine-Tuning geeignet
```

Diese Frage hat praktische Implikationen:

- Wenn ML besser transferiert → Empfehlung für ressourcenarme Städte
- Wenn NN besser transferiert → Deep Learning lohnt sich auch für Transfer

---

## Transfer-Metriken

### Absolute und Relative Metriken

| Metrik            | Berechnung                                 | Interpretation                                       |
| ----------------- | ------------------------------------------ | ---------------------------------------------------- |
| **Absolute Drop** | F1_Berlin - F1_Leipzig                     | Direkte Performance-Differenz in Prozentpunkten      |
| **Relative Drop** | (F1_Berlin - F1_Leipzig) / F1_Berlin × 100 | Prozentualer Verlust bezogen auf Ausgangsperformance |

**Warum beide?**

```
Beispiel 1:
├── Berlin F1: 0.80
├── Leipzig F1: 0.72
├── Absolute Drop: 0.08 (8 Prozentpunkte)
└── Relative Drop: 10%

Beispiel 2:
├── Berlin F1: 0.50
├── Leipzig F1: 0.42
├── Absolute Drop: 0.08 (8 Prozentpunkte)  ← Gleich!
└── Relative Drop: 16%  ← Deutlich höher!

Interpretation:
• Absolute Drop zeigt absolute Verschlechterung
• Relative Drop normalisiert für Ausgangsperformance
• Beide zusammen geben vollständiges Bild
```

### Per-Genus Transfer-Robustheit

Für jede Baumgattung berechnen wir:

```
Robustheit(Genus) = |F1_Berlin(Genus) - F1_Leipzig(Genus)| / F1_Berlin(Genus) × 100
```

**Klassifikation:**

| Kategorie  | Relativer Drop | Interpretation                |
| ---------- | -------------- | ----------------------------- |
| **Robust** | < 5%           | Genus transferiert exzellent  |
| **Medium** | 5% - 15%       | Genus transferiert akzeptabel |
| **Poor**   | > 15%          | Genus hat Transfer-Probleme   |

**Wissenschaftliche Basis der Schwellenwerte:**

- **5%:** Typische Varianz zwischen Trainingsläufen; <5% ist praktisch identisch
- **15%:** Entspricht signifikantem Unterschied; User würde Qualitätsverlust bemerken
- Basiert auf Praktiker-Feedback und Domain-Expertise

### Per-Genus Analyse: Hypothesenbildung

Die Per-Genus-Analyse soll Hypothesen generieren:

| Mögliches Muster                      | Hypothese                                     | Praktische Implikation             |
| ------------------------------------- | --------------------------------------------- | ---------------------------------- |
| Acer transferiert gut, Tilia schlecht | Acer-Phenologie konsistenter zwischen Städten | Bei Tilia lokale Daten wichtiger   |
| Seltene Genera transferieren schlecht | Zu wenig Trainingsdaten für Generalisierung   | Fokus auf häufige Genera           |
| Alle Genera transferieren schlecht    | Fundamentale Unterschiede Berlin/Leipzig      | Transfer-Learning weniger geeignet |

---

## Experimentelles Design

### Datenvorbereitung

```
1. Lade Leipzig Test Set
   └── Preprocessing: Gleicher Scaler wie Berlin (wichtig!)

2. Lade beide Berlin-Champions
   └── ML: berlin_ml_champion.pkl
   └── NN: berlin_nn_champion.pt

3. Predict auf Leipzig
   └── Keine Anpassung, keine Threshold-Optimierung
```

**Kritisch: Gleicher Scaler**

```python
# RICHTIG:
scaler = load("berlin_scaler.pkl")  # Auf Berlin gefittet
X_leipzig_scaled = scaler.transform(X_leipzig)  # Nur transform!

# FALSCH:
scaler = StandardScaler()
X_leipzig_scaled = scaler.fit_transform(X_leipzig)  # Data Leakage!
```

**Begründung:** Wenn wir den Scaler auf Leipzig fitten, adaptieren wir implizit an die Zieldomäne. Das ist keine echte Zero-Shot Evaluation.

### Metriken berechnen

Für jedes Modell (ML und NN):

```python
metrics = {
    "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    "macro_f1": f1_score(y_true, y_pred, average="macro"),
    "accuracy": accuracy_score(y_true, y_pred),
    "per_genus_f1": classification_report(y_true, y_pred, output_dict=True),
    "confusion_matrix": confusion_matrix(y_true, y_pred)
}
```

### Transfer-Gap berechnen

```python
def compute_transfer_gap(berlin_metrics, leipzig_metrics):
    berlin_f1 = berlin_metrics["weighted_f1"]
    leipzig_f1 = leipzig_metrics["weighted_f1"]

    absolute_drop = berlin_f1 - leipzig_f1
    relative_drop = (absolute_drop / berlin_f1) * 100

    return {
        "absolute_drop": absolute_drop,
        "relative_drop_pct": relative_drop
    }
```

### Konfidenzintervalle

```python
def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=0.95):
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)

    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return lower, upper
```

---

## Visualisierungen

### 1. Transfer Comparison Bar Chart

```
        Berlin      Leipzig
        ┌────┐
        │    │      ┌────┐
        │    │      │    │
        │ ML │      │ ML │
 F1     │    │      │    │
0.6 ────┼────┼──────┼────┼─────
        │    │      │    │
        │    │      │    │
        └────┘      └────┘

        ┌────┐
        │    │      ┌────┐
        │ NN │      │ NN │
        │    │      │    │
        └────┘      └────┘
```

**Zeigt auf einen Blick:** Welches Modell wie viel verliert

### 2. Confusion Matrix Comparison

```
        BERLIN                    LEIPZIG
    ┌─────────────────┐       ┌─────────────────┐
    │ A  B  C  D  E  │       │ A  B  C  D  E  │
  A │ 45 2  1  0  2  │     A │ 38 5  3  1  3  │
  B │ 3  40 5  0  2  │     B │ 5  32 8  2  3  │
  C │ 2  4  38 4  2  │     C │ 4  6  30 6  4  │
  D │ 1  1  3  42 3  │     D │ 2  3  5  35 5  │
  E │ 2  3  2  1  42 │     E │ 3  4  3  2  38 │
    └─────────────────┘       └─────────────────┘
```

**Zeigt:** Systematische Verschiebungen (z.B. Verwechslung A↔B wird häufiger)

### 3. Per-Genus Transfer Robustness

```
Genus           Berlin F1    Leipzig F1    Drop    Robustness
──────────────────────────────────────────────────────────────
Acer            0.72         0.69          4%      ████████████ Robust
Tilia           0.68         0.61          10%     ████████░░░░ Medium
Quercus         0.65         0.52          20%     █████░░░░░░░ Poor
Platanus        0.70         0.58          17%     █████░░░░░░░ Poor
...
```

---

## Hypothesen und erwartete Ergebnisse

### H1: Transfer-Gap ist signifikant

**Erwartung:** Leipzig F1 wird 10-30% unter Berlin F1 liegen

**Begründung:**

- Unterschiedliche Klimabedingungen → andere Phenologie
- Unterschiedliche Genus-Verteilungen
- Potentiell andere Baumalter/Vitalität

### H2: ML transferiert robuster als NN

**Erwartung:** XGBoost/RF hat kleineren relativen Drop als 1D-CNN/TabNet

**Begründung:**

- NNs können stärker auf stadtspezifische Muster overfitten
- Random Forest aggregiert über viele einfache Entscheidungen → robuster

**Aber:** Wenn die Daten sauber und Features generell sind, könnte NN auch gleich/besser transferieren.

### H3: Strukturell unterschiedliche Genera transferieren schlecht

**Erwartung:** Genera mit großer morphologischer Varianz (z.B. Acer, Prunus) transferieren schlechter

**Begründung:**

- Innerhalb eines Genus können Arten stark variieren
- Berliner "Acer" könnte andere Unterarten enthalten als Leipziger "Acer"

---

## Best Transfer Model Selektion

### Entscheidungslogik

```
if ML_relative_drop < NN_relative_drop - 5%:
    best_transfer = "ML"
    reasoning = "ML transferiert signifikant besser"
elif NN_relative_drop < ML_relative_drop - 5%:
    best_transfer = "NN"
    reasoning = "NN transferiert signifikant besser"
else:
    # Beide ähnlich → wähle bessere Absolute Performance
    if ML_leipzig_f1 > NN_leipzig_f1:
        best_transfer = "ML"
    else:
        best_transfer = "NN"
    reasoning = "Kein signifikanter Transfer-Unterschied, gewählt nach Leipzig F1"
```

**Schwellenwert 5%:** Kleinere Unterschiede sind praktisch irrelevant und könnten durch Zufall entstanden sein.

---

## Outputs

### transfer_evaluation.json

```json
{
    "timestamp": "2026-02-03T14:30:00Z",
    "source_city": "Berlin",
    "target_city": "Leipzig",
    "models": {
        "ml_champion": {
            "name": "xgboost",
            "source_f1": 0.62,
            "source_f1_ci": [0.59, 0.65],
            "target_f1": 0.48,
            "target_f1_ci": [0.44, 0.52],
            "absolute_drop": 0.14,
            "relative_drop_pct": 22.6,
            "per_genus_robustness": {
                "Acer": {"drop_pct": 8, "category": "medium"},
                "Tilia": {"drop_pct": 18, "category": "poor"},
                ...
            }
        },
        "nn_champion": {
            ...
        }
    },
    "best_transfer_model": "ml_champion",
    "selection_reasoning": "ML hat 5% geringeren relativen Drop"
}
```

### Visualisierungen

| Datei                                   | Inhalt                                  |
| --------------------------------------- | --------------------------------------- |
| transfer_comparison.png                 | Berlin vs. Leipzig F1 für beide Modelle |
| confusion_comparison_berlin_leipzig.png | Side-by-side Confusion Matrices         |
| per_genus_transfer_robustness.png       | Robustheits-Klassifikation pro Genus    |

---

## Praktische Implikationen

### Für Städte ohne Trainingsdaten

| Transfer-Drop | Empfehlung                                         |
| ------------- | -------------------------------------------------- |
| < 10%         | Zero-Shot nutzbar, Fine-Tuning optional            |
| 10-20%        | Zero-Shot möglich, Fine-Tuning empfohlen           |
| 20-30%        | Zero-Shot als Startpunkt, Fine-Tuning notwendig    |
| > 30%         | Transfer-Learning fragwürdig, From-Scratch erwägen |

### Für diese Arbeit

Der Transfer-Gap bestimmt, wie wichtig Fine-Tuning ist:

- Kleiner Gap → Fokus auf Effizienz (wenig Daten reichen)
- Großer Gap → Fokus auf Effektivität (wie viel Daten für Wiederherstellung?)

---

_Letzte Aktualisierung: 2026-02-03_
