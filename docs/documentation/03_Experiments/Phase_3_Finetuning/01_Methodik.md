# Phase 3.5: Fine-Tuning

## Einleitung

Das Fine-Tuning beantwortet die praktisch relevanteste Frage:

> **Wie viele lokale Trainingsdaten benötigt eine Stadt, um von einem transferierten Modell zu einem akzeptabel performenden lokalen Modell zu gelangen?**

Diese Information hat direkte budgetäre Implikationen: Lohnt es sich, 100 Bäume manuell zu labeln, oder braucht man 1000?

---

## Forschungsfragen

1. **Sample Efficiency:** Wie viele Leipzig-Daten sind nötig, um X% der From-Scratch Performance zu erreichen?
2. **Transfer-Vorteil:** Ist Transfer + Fine-Tuning besser als From-Scratch Training mit gleichen Daten?
3. **Konvergenzpunkt:** Ab welcher Datenmenge ist Transfer-Vorteil vernachlässigbar?

---

## Methodische Begründungen

### Warum nur bestes Transfer-Modell?

```
Option A: Beide Champions fine-tunen
├── Pro: Vollständiger Vergleich ML vs. NN
├── Contra: Doppelter Aufwand (4 Fraktionen × 2 Modelle = 8 Experimente)
└── Entscheidung: Nicht gewählt

Option B: Nur bestes Transfer-Modell (gewählt)
├── Pro: Fokussiert, effizienter
├── Begründung: Wenn ML besser transferiert, ist Fine-Tuning von NN weniger relevant
└── Das Transfer-Modell mit besserem Zero-Shot ist wahrscheinlich auch nach Fine-Tuning besser
```

### Warum diese Fraktionen: 10%, 25%, 50%, 100%?

```
Fraktion    Samples (bei ~5000 Leipzig)    Repräsentiert
──────────────────────────────────────────────────────────────
10%         ~500                           Minimal-Investment
25%         ~1250                          Moderate Investment
50%         ~2500                          Substantial Investment
100%        ~5000                          Vollständige Nutzung
```

**Begründung der Auswahl:**

- **Logarithmische Skalierung:** 10→25→50→100 zeigt Diminishing Returns
- **Praktische Relevanz:** 10% = "Können wir mit wenig Aufwand ausreichend verbessern?"
- **Interpolierbar:** Zwischen 25% und 50% kann linear interpoliert werden

### Warum stratifizierte Subsets?

```python
# RICHTIG: Stratified Sampling
from sklearn.model_selection import train_test_split
X_10pct, _, y_10pct, _ = train_test_split(
    X_finetune, y_finetune,
    train_size=0.1,
    stratify=y_finetune,  # Klassenverteilung erhalten
    random_state=42
)

# FALSCH: Random Sampling
X_10pct = X_finetune[:500]  # Könnte bestimmte Klassen ausschließen
```

**Begründung:**

- Bei 10% könnte zufällig eine seltene Genus-Klasse komplett fehlen
- Stratified garantiert proportionale Repräsentation aller Klassen
- Ermöglicht fairen Vergleich zwischen Fraktionen

---

## Fine-Tuning Strategien

### Für ML (XGBoost/Random Forest)

**Methode: Warm-Start / Continue Training**

```python
# XGBoost: Zusätzliche Bäume trainieren
finetuned_model = xgb.train(
    params=pretrained_params,
    dtrain=DMatrix(X_finetune, y_finetune),
    num_boost_round=100,  # Zusätzliche Bäume
    xgb_model=pretrained_model  # Start von Berlin-Modell
)
```

**Begründung:**

- XGBoost/RF hat keine "Layers" zum Freezing
- Warm-Start nutzt Berlin-Wissen als Initialisierung
- Zusätzliche Bäume spezialisieren sich auf Leipzig-Muster

### Für NN (1D-CNN/TabNet)

**Methode: Full Fine-Tune mit reduzierter Learning Rate**

```python
# Learning Rate auf 10% der Original-LR reduzieren
optimizer = Adam(pretrained_model.parameters(), lr=original_lr * 0.1)

# Training auf Leipzig-Daten
for epoch in range(50):
    train_epoch(pretrained_model, X_finetune, y_finetune)
```

**Begründung:**

- Niedrige LR verhindert "Vergessen" des Berlin-Wissens
- 0.1× ist Standardwert in Transfer-Learning-Literatur
- Alle Weights werden angepasst (Full Fine-Tune)

**Warum nicht Freeze Layers?**

```
Freeze Early Layers:
├── Sinnvoll wenn: Source und Target sehr ähnlich
├── Unser Fall: Berlin und Leipzig sind unterschiedlich genug
└── Risk: Frühe Layers könnten stadtspezifisch sein

→ Full Fine-Tune ist konservativer und robuster
```

### Class Weighting bei Fine-Tuning

**Entscheidung:** Class Weights auf Leipzig-Verteilung neu berechnen

```python
# Leipzig-Klassenverteilung verwenden
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_finetune),
    y=y_finetune
)
```

**Begründung:**

- Fine-Tuning optimiert für Leipzig, nicht Berlin
- Leipzig hat andere Genus-Verteilung
- `balanced` behandelt alle Klassen gleich wichtig

---

## From-Scratch Baseline

### Definition

Ein Modell, das **komplett neu** auf Leipzig-Daten trainiert wird — ohne jegliches Berlin-Vorwissen.

```
From-Scratch Baseline:
├── Training: 100% Leipzig Finetune
├── Validation: Aus Leipzig Finetune (z.B. 80/20 Split)
├── Test: Leipzig Test
└── Kein Transfer, kein Pretraining
```

### Zweck

| Vergleich                       | Frage                                            |
| ------------------------------- | ------------------------------------------------ |
| Zero-Shot vs. From-Scratch      | Ist Transfer überhaupt sinnvoll?                 |
| Fine-Tune 100% vs. From-Scratch | Verbessert Pretraining die 100%-Performance?     |
| Fine-Tune X% vs. From-Scratch   | Ab wann ist Fine-Tuning besser als From-Scratch? |

### Erwartetes Ergebnis

```
Typische Sample Efficiency Curve:

F1
│
│                        ┌──────── From-Scratch 100%
│                     .-'
│                  .-'
│               .-'          ┌──── Fine-Tune 100%
│            .-'          .-'
│         .-'          .-'
│      .-' Fine-Tune.-'
│   .-'         .-'
│.-' Zero-Shot-'
└────────────────────────────────── Fraktion %
    10%    25%    50%    100%

Erwartung:
• Fine-Tune startet über From-Scratch (Transfer-Vorteil)
• Kurven konvergieren bei 100%
• Fine-Tune erreicht From-Scratch schneller
```

---

## Statistische Signifikanztests

### McNemar Test

Der McNemar Test vergleicht, ob zwei Klassifikatoren **signifikant unterschiedliche Fehler** machen.

```
                    Modell B
                   Wrong  Right
Modell A  Wrong     a       b
          Right     c       d

McNemar Statistik: χ² = (b - c)² / (b + c)
```

**Anwendung:**

| Vergleich                       | Frage                                                |
| ------------------------------- | ---------------------------------------------------- |
| Zero-Shot vs. Fine-Tune 10%     | Verbessert 10% signifikant?                          |
| Fine-Tune 25% vs. 50%           | Lohnt sich Verdopplung der Daten?                    |
| Fine-Tune 100% vs. From-Scratch | Ist Transfer bei voller Datenmenge noch vorteilhaft? |

**Signifikanzniveau:** α = 0.05

**Interpretation:**

- p < 0.05: Unterschied ist statistisch signifikant
- p ≥ 0.05: Unterschied könnte zufällig sein

### Konfidenzintervalle

Für jede Fraktion: Bootstrap CI für F1

```python
def bootstrap_ci(y_true, y_pred, n_bootstrap=1000):
    f1_scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        f1_scores.append(f1_score(y_true[idx], y_pred[idx], average='weighted'))
    return np.percentile(f1_scores, [2.5, 97.5])
```

---

## Effizienz-Metriken

### Fraction to Match Scratch

**Definition:** Minimale Fine-Tuning-Fraktion, bei der Fine-Tune F1 ≥ From-Scratch F1

```
Beispiel:
├── From-Scratch 100% F1: 0.52
├── Fine-Tune 25% F1: 0.54  ← Bereits besser!
└── Fraction to Match: ≤ 25%

Interpretation: Mit nur 25% der Daten + Transfer erreicht man
                mehr als mit 100% der Daten ohne Transfer
```

### Fraction to 90% of Scratch

**Definition:** Fraktion, bei der Fine-Tune 90% der From-Scratch Performance erreicht

```
Beispiel:
├── From-Scratch 100% F1: 0.52
├── 90% davon: 0.47
├── Fine-Tune 10% F1: 0.45  ← Unter 90%
├── Fine-Tune 25% F1: 0.49  ← Über 90%
└── Fraction to 90%: ~20% (interpoliert)
```

**Praktische Bedeutung:**

- Zeigt, wie effizient Transfer-Learning ist
- 90%-Schwelle = "praktisch akzeptable" Performance
- Niedriger Wert = Transfer ist sehr effizient

---

## Experimenteller Ablauf

```
1. Lade Best Transfer Model (aus Phase 3.4)
   └── ML oder NN, basierend auf transfer_evaluation.json

2. Erstelle stratifizierte Subsets
   └── 10%, 25%, 50%, 100% von Leipzig Finetune

3. Für jede Fraktion:
   ├── Fine-Tune Modell auf Subset
   ├── Evaluiere auf Leipzig Test
   └── Speichere Metriken

4. From-Scratch Baseline
   ├── Trainiere neues Modell auf 100% Leipzig Finetune
   └── Evaluiere auf Leipzig Test

5. Statistische Tests
   ├── McNemar: Paarweise Vergleiche
   └── Konfidenzintervalle: Bootstrap

6. Effizienz-Metriken berechnen
   ├── Fraction to Match Scratch
   └── Fraction to 90% of Scratch

7. Visualisierungen erstellen
```

---

## Outputs

### finetuning_curve.json

```json
{
    "timestamp": "2026-02-03T16:00:00Z",
    "model": "xgboost",
    "finetuning_method": "warm_start_100_trees",
    "results": [
        {
            "fraction": 0.0,
            "n_samples": 0,
            "f1": 0.48,
            "f1_ci_lower": 0.44,
            "f1_ci_upper": 0.52,
            "pct_of_from_scratch": 92.3
        },
        {
            "fraction": 0.1,
            "n_samples": 500,
            "f1": 0.51,
            "f1_ci_lower": 0.47,
            "f1_ci_upper": 0.55,
            "pct_of_from_scratch": 98.1
        },
        ...
    ],
    "baselines": {
        "zero_shot_f1": 0.48,
        "from_scratch_f1": 0.52
    },
    "efficiency_metrics": {
        "fraction_to_match_scratch": 0.25,
        "fraction_to_90pct_scratch": 0.08
    },
    "significance_tests": [
        {
            "comparison": "zero_shot_vs_10pct",
            "mcnemar_statistic": 12.4,
            "p_value": 0.0004,
            "significant": true
        },
        ...
    ]
}
```

### Visualisierungen

| Datei                       | Inhalt                        |
| --------------------------- | ----------------------------- |
| finetuning_curve.png        | F1 vs. Fraktion mit Baselines |
| finetuning_vs_baselines.png | Vergleich aller Varianten     |

#### Sample Efficiency Curve (finetuning_curve.png)

```
F1
0.60 ┤
     │                              ┌─── From-Scratch
0.55 ┤                          ••••○
     │                      •••'
0.50 ┤                  •••'
     │              •••'────────────┐
     │          ••○'                │ Fine-Tune
0.45 ┤      •••'                    │
     │  ○••'                        │
0.40 ┤○─────────────────────────────┘ Zero-Shot
     │
     └──┬────────┬────────┬────────┬──
       0%      25%      50%     100%
              Fine-Tuning Fraktion
```

---

## Erwartete Ergebnisse

### Hypothesen

| Hypothese                 | Erwartung                          | Begründung                             |
| ------------------------- | ---------------------------------- | -------------------------------------- |
| H1: Fine-Tune > Zero-Shot | Bereits 10% verbessert signifikant | Jegliche lokale Daten helfen           |
| H2: Diminishing Returns   | 25%→50% bringt weniger als 10%→25% | Marginaler Nutzen sinkt                |
| H3: Transfer = Vorteil    | Fine-Tune 50% ≈ From-Scratch 100%  | Transfer spart ~50% Daten              |
| H4: Konvergenz            | Fine-Tune 100% ≈ From-Scratch 100% | Bei voller Datenmenge gleicht sich aus |

### Praktische Interpretationsrahmen

| Ergebnis                            | Praktische Empfehlung                       |
| ----------------------------------- | ------------------------------------------- |
| 10% reicht für 90% Performance      | Minimal-Investment genügt                   |
| 25% nötig für 90%                   | Moderater Labelaufwand                      |
| 50%+ nötig für 90%                  | Substantieller Aufwand, aber Transfer lohnt |
| Kein Vorteil gegenüber From-Scratch | Transfer für diese Städte nicht empfohlen   |

---

## Praktische Implikationen

### Für Städte ohne Trainingsdaten

```
Empfohlener Workflow:
1. Berlin-Modell Zero-Shot anwenden → Baseline messen
2. ~10% der Bäume manuell labeln → Fine-Tune → Messen
3. Bei unzureichender Performance → 25% labeln → Fine-Tune
4. Repeat bis akzeptable Performance erreicht
```

### Kostenschätzung (hypothetisch)

| Fraktion       | Samples | Labelaufwand (h) | Performance-Gewinn |
| -------------- | ------- | ---------------- | ------------------ |
| 0% (Zero-Shot) | 0       | 0                | Baseline           |
| 10%            | 500     | ~10h             | +5% F1             |
| 25%            | 1250    | ~25h             | +2% F1             |
| 50%            | 2500    | ~50h             | +1% F1             |

→ **Kosten-Nutzen-Analyse** für kommunale Entscheidungsträger

---

_Letzte Aktualisierung: 2026-02-03_
