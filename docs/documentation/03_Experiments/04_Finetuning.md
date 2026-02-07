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

### Warum beide Champions fine-tunen?

```
Option A: Beide Champions fine-tunen (gewählt)
├── Pro: Vollständiger ML vs. NN Vergleich über alle Fraktionen
├── Pro: Zeigt, ob ML/NN unterschiedlich von lokalen Daten profitieren
├── Pro: Sample-Efficiency-Kurven beider Paradigmen direkt vergleichbar
├── Aufwand: 4 Fraktionen × 2 Modelle = 8 Experimente (vertretbar)
└── Kernargument: Zero-Shot-Ranking ≠ Fine-Tuning-Ranking
    └── Ein Modell mit schlechterem Transfer kann nach Fine-Tuning besser sein

Option B: Nur bestes Transfer-Modell
├── Pro: Weniger Aufwand
├── Contra: Verliert ML-vs-NN Fine-Tuning-Vergleich
└── Entscheidung: Nicht gewählt — der Vergleich ist wissenschaftlich zu wertvoll
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

## Scaler-Strategie beim Fine-Tuning

### Entscheidung

| Szenario                          | Scaler                          | Begründung                                                                                           |
| --------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Fine-Tuning** (Transfer-Modell) | Berlin-Scaler beibehalten       | Modell-Weights erwarten Berlin-skalierte Features; nur Weights werden angepasst, nicht Preprocessing |
| **From-Scratch Baseline**         | Neuer Scaler auf Leipzig fitten | Kein Vorwissen, eigenständiges Modell                                                                |

### Warum Berlin-Scaler beim Fine-Tuning?

```python
# RICHTIG: Berlin-Scaler beibehalten
scaler = load("berlin_scaler.pkl")        # Auf Berlin Train gefittet
X_finetune_scaled = scaler.transform(X_finetune)  # Nur transform!
X_test_scaled = scaler.transform(X_test)           # Konsistent

# FALSCH: Neuen Scaler auf Leipzig fitten
scaler = StandardScaler()
X_finetune_scaled = scaler.fit_transform(X_finetune)  # Verändert Feature-Raum!
```

**Begründung:**

- Die Modell-Weights (Baumstrukturen bei XGBoost, Neuronale Gewichte bei NN) wurden auf Berlin-skalierte Werte trainiert
- Ein neuer Scaler würde die Feature-Verteilung verschieben → Modell-Wissen wird inkonsistent
- Gleicher Scaler macht Zero-Shot und Fine-Tuning direkt vergleichbar
- Die From-Scratch Baseline nutzt einen eigenen Leipzig-Scaler, weil dort kein Vorwissen existiert

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

### Power-Law Fit (Improvement 6)

**Zweck:** Quantifizierung der Sample-Efficiency-Kurve durch analytisches Modell.

Sample-Efficiency-Kurven folgen häufig einem **Power-Law** (Potenzgesetz):

```
Performance = a × N^b
```

wobei:
- **N**: Anzahl Fine-Tuning Samples
- **a**: Skalierungsfaktor
- **b**: Sample-Efficiency-Exponent (0 < b < 1)

**Implementierung:**

```python
from scipy.optimize import curve_fit

def power_law(x, a, b):
    return a * np.power(x, b)

# Fit auf Fine-Tuning Datenpunkte
fractions = [0.1, 0.25, 0.5, 1.0]
sample_counts = [500, 1250, 2500, 5000]
f1_scores = [0.51, 0.54, 0.56, 0.57]

params, _ = curve_fit(power_law, sample_counts, f1_scores)
a, b = params
```

**Anwendungen:**

1. **Extrapolation zu 95% Recovery:**
   ```python
   # Wie viele Samples für 95% der From-Scratch Performance?
   target_f1 = from_scratch_f1 * 0.95
   n_samples_95pct = ((target_f1 - zero_shot_f1) / a) ** (1/b)
   ```

2. **Visualisierung:**
   - Gestrichelte Kurve: Power-Law Fit
   - Punkte: Empirische Messungen (10%, 25%, 50%, 100%)
   - Extrapolation: Gestrichelte Linie über 100% hinaus

**Interpretation des Exponenten b:**

| Exponent b | Interpretation                                  |
| ---------- | ----------------------------------------------- |
| b > 0.5    | Starke Sample-Efficiency (rapide Verbesserung)  |
| b ≈ 0.3    | Moderate Sample-Efficiency                      |
| b < 0.2    | Schwache Sample-Efficiency (viele Daten nötig)  |

**Literatur-Vergleich (Improvement 6):**

| Studie                | Domain            | Label Savings | b-Exponent |
| --------------------- | ----------------- | ------------- | ---------- |
| Tong et al. (2019)    | Remote Sensing    | 50-70%        | ~0.35      |
| Dieses Projekt        | Tree Transfer     | TBD           | TBD        |

**HINWEIS:** Wenn b < 0.2, deutet dies auf limitierte Transferierbarkeit zwischen Berlin und Leipzig hin.

---

## Experimenteller Ablauf

```
1. Lade beide Transfer-Champions (aus Phase 3.3)
   ├── ML-Champion (z.B. XGBoost)
   └── NN-Champion (z.B. TabNet)

2. Erstelle stratifizierte Subsets
   └── 10%, 25%, 50%, 100% von Leipzig Finetune
   └── Gleiche Subsets für beide Modelle (fairer Vergleich)

3. Für jede Fraktion × jedes Modell:
   ├── ML: Continue Training / Warm Start
   ├── NN: Full Fine-Tune mit 0.1× LR
   ├── Evaluiere auf Leipzig Test
   └── Speichere Metriken

4. From-Scratch Baselines
   ├── ML von Grund auf auf 100% Leipzig Finetune
   ├── NN von Grund auf auf 100% Leipzig Finetune
   └── Evaluiere beide auf Leipzig Test

5. Statistische Tests
   ├── McNemar: Paarweise Vergleiche
   └── Konfidenzintervalle: Bootstrap

6. Effizienz-Metriken berechnen
   ├── Fraction to Match Scratch (je Modell)
   ├── Fraction to 90% of Scratch (je Modell)
   └── **Power-Law Fit (Imp 6):**
       ├── Fit y = a × x^b auf empirische Datenpunkte
       ├── Extrapoliere zu 95% Recovery Point
       └── Vergleich mit Literatur (Tong et al. 2019: b ≈ 0.35)

7. Visualisierungen erstellen
   ├── Sample-Efficiency-Kurven: ML + NN auf gleichem Plot
   └── **Power-Law Fit Curve (Imp 6)** mit Extrapolation (gestrichelt)
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
    "power_law_fit": {
        "a": 0.28,
        "b": 0.32,
        "r_squared": 0.97,
        "interpretation": "moderate sample efficiency (b=0.32)"
    },
    "recovery_points": {
        "n_samples_90pct": 387,
        "n_samples_95pct": 892,
        "label_savings_vs_scratch": "82.2%"
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

| Datei                                   | Inhalt                                                          |
| --------------------------------------- | --------------------------------------------------------------- |
| finetuning_curve.png                    | F1 vs. Fraktion mit Zero-Shot + From-Scratch Baselines          |
| **power_law_fit.png (Imp 6)**           | Power-Law Kurve mit Extrapolation zu 95% Recovery (gestrichelt) |
| finetuning_vs_baselines.png             | Vergleich aller Varianten                                       |
| finetuning_per_genus_recovery.png       | Heatmap: Pro-Gattung F1 bei jeder Fraktion (deutsch)            |
| finetuning_ml_vs_nn_comparison.png      | ML vs. NN Sample-Efficiency-Kurven                              |
| finetuning_significance_matrix.png      | McNemar p-Werte über alle Vergleichspaare                       |

**Hinweis:** Alle Genus-Labels nutzen **deutsche Gattungsnamen**. Die Pro-Gattung
Recovery-Heatmap ermöglicht direkte Aussagen wie "Linde erholt sich bereits bei 10% Fine-Tuning
auf 90% der From-Scratch Performance, Eiche erst bei 50%."

#### Sample Efficiency Curve mit Power-Law Fit (Imp 6)

```
F1
0.60 ┤
     │                              ┌─── From-Scratch
0.55 ┤                          ••••○
     │                      •••'
0.50 ┤                  •••'┈┈┈┈┈┈┈┈┐ Power-Law
     │              •••'            │ Extrapolation
     │          ••○'                │ (gestrichelt)
0.45 ┤      •••'┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┐  │
     │  ○••'                      │  │ Fine-Tune (Punkte)
0.40 ┤○───────────────────────────┴──┘ Zero-Shot
     │        ↑
     │        └─ 95% Recovery Point (aus Power-Law)
     └──┬────────┬────────┬────────┬────────┬──
       0%      25%      50%     100%    150%
              Fine-Tuning Fraktion

Legende:
• ○ = Empirische Messungen (10%, 25%, 50%, 100%)
┈┈┈ = Power-Law Fit (y = a × x^b)
```

**Beschriftung:** Visualisierung zeigt empirische Datenpunkte und gefittete Power-Law Kurve mit Extrapolation zu 95% Recovery Point.

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

_Letzte Aktualisierung: 2026-02-06_
