# Phase 3.1-3.3: Berlin-Optimierung

## Einleitung

Die Berlin-Optimierung bildet das Fundament aller nachfolgenden Experimente. In dieser Phase etablieren wir die **Upper Bound** — die bestmögliche Performance, die ein Modell auf Berliner Daten erreichen kann. Diese dient als Referenz für die Transfer- und Fine-Tuning-Evaluation.

---

## Forschungsfragen

1. **Setup:** Welche CHM-Normalisierung und Feature-Anzahl sind optimal für die Klassifikation?
2. **Algorithmen:** Welcher ML- und welcher NN-Algorithmus performt am besten?
3. **Optimierung:** Wie viel Verbesserung bringt Hyperparameter-Tuning gegenüber Default-Parametern?

---

## Methodische Begründungen

### Warum Setup-Entscheidungen vor Algorithmenvergleich?

Die Reihenfolge "Setup fixieren → dann Algorithmen vergleichen" folgt dem Prinzip der **kontrollierten Variation**:

```
Option A (gewählt):                Option B (verworfen):
──────────────────────             ─────────────────────
1. CHM-Strategie fixieren          1. Alle Algorithmen mit
2. Features selektieren               allen Setups testen
3. Dann alle Algorithmen           2. Kombinatorische Explosion:
   auf GLEICHEM Setup                 5 CHM × 4 Features × 4 Algos
   vergleichen                        = 80+ Experimente
```

**Begründung:**

- Faire Vergleichbarkeit: Alle Algorithmen nutzen identische Features
- Effizienz: Weniger Experimente nötig
- Interpretierbarkeit: Unterschiede sind eindeutig dem Algorithmus zuzuordnen

### Warum Random Forest als Baseline-Algorithmus für Setup?

Für die Setup-Entscheidungen (CHM, Features) verwenden wir **Random Forest mit Default-Parametern** als stabilen Baseline:

| Eigenschaft          | Bedeutung für Setup-Experimente           |
| -------------------- | ----------------------------------------- |
| Geringes Overfitting | Verzerrte Entscheidungen werden vermieden |
| Deterministisch      | Reproduzierbare Ergebnisse                |
| Feature Importance   | Direkt verfügbar für Feature-Selektion    |
| Schnelles Training   | Ermöglicht 5-Fold CV                      |

Ein optimiertes XGBoost oder NN könnte durch Overfitting auf bestimmte Features die CHM-Entscheidung verzerren.

---

## Phase 3.1: Setup-Fixierung

### CHM-Ablation (exp_08)

#### Motivation

Das Canopy Height Model (CHM) liefert strukturelle Information über Baumhöhe und Kronenstruktur. Es ist jedoch unklar:

1. **Verbessert CHM die Klassifikation?** — Wenn ja, lohnt sich die zusätzliche Komplexität
2. **Welche Normalisierung ist optimal?** — Raw, Z-Score, Perzentile, oder beide?
3. **Schadet CHM dem Transfer?** — Wenn CHM stadtspezifisch ist, könnte es Overfitting auf Berlin verursachen

#### Experimentelles Design

| Variante   | CHM-Features                             | Sentinel-2 Features |
| ---------- | ---------------------------------------- | ------------------- |
| no_chm     | Keine                                    | Alle S2 Features    |
| raw        | chm_mean, chm_std, etc. (unnormalisiert) | Alle S2 Features    |
| zscore     | chm_mean_zscore, etc.                    | Alle S2 Features    |
| percentile | chm_mean_pct, etc.                       | Alle S2 Features    |
| both       | zscore + percentile                      | Alle S2 Features    |

#### Entscheidungslogik

```
                    ┌─────────────────────────────────┐
                    │ CHM Importance > 25%?           │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Ja                      │ Nein
                    ▼                         ▼
         ┌──────────────────┐    ┌─────────────────────────┐
         │ → Kein CHM       │    │ Best CHM F1 - No CHM F1 │
         │ (Overfitting-    │    │        > 0.03?          │
         │  Risiko)         │    └───────────┬─────────────┘
         └──────────────────┘                │
                                ┌────────────┴────────────┐
                                │ Ja                      │ Nein
                                ▼                         ▼
                    ┌──────────────────────┐  ┌──────────────────┐
                    │ Transfer-Drop mit    │  │ → Kein CHM       │
                    │ CHM > 15%?           │  │ (Marginaler      │
                    └──────────┬───────────┘  │  Gewinn)         │
                               │              └──────────────────┘
                  ┌────────────┴────────────┐
                  │ Ja                      │ Nein
                  ▼                         ▼
       ┌──────────────────┐    ┌──────────────────────┐
       │ → Kein CHM       │    │ → Beste CHM-Variante │
       │ (Schadet         │    │    verwenden         │
       │  Transfer)       │    └──────────────────────┘
       └──────────────────┘
```

#### Wissenschaftliche Begründung der Schwellenwerte

| Schwellenwert        | Begründung                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------- |
| CHM Importance > 25% | Wenn ein einzelnes Feature-Set >25% der Vorhersagekraft trägt, besteht Overfitting-Risiko   |
| F1-Gewinn < 0.03     | Unterschiede <3 Prozentpunkte sind praktisch insignifikant und innerhalb der Varianz        |
| Transfer-Drop > 15%  | 15% relativer Drop klassifiziert als "poor" Transfer, inakzeptabel für praktische Anwendung |

### Feature-Reduktion (exp_09)

#### Motivation

Die Feature-Anzahl beeinflusst:

1. **Performance:** Zu viele Features → Overfitting; zu wenige → Underfitting
2. **Interpretierbarkeit:** Weniger Features sind verständlicher
3. **Trainingszeit:** Weniger Features = schnellere Modelle
4. **Transfer-Robustheit:** Generelle Features transferieren besser als spezifische

#### Methodik: Importance-basierte Selektion

```
1. Trainiere RF mit allen Features
   └── Extrahiere Gain-basierte Importance

2. Ranke Features nach Importance
   └── Erstelle Subsets: Top-30, Top-50, Top-80, Alle

3. Evaluiere jedes Subset mit 5-Fold CV
   └── Messe: F1, Trainingszeit

4. Pareto-Analyse
   └── Finde Kniepunkt: Minimale Features bei maximaler F1
```

#### Entscheidungslogik

**Regel:** Wähle kleinstes k, sodass F1(Top-k) ≥ F1(Alle) - 0.01

**Begründung des 1%-Schwellenwerts:**

- 1% F1-Verlust ist praktisch irrelevant
- Entspricht typischer Varianz zwischen Trainingsläufen
- Ermöglicht signifikante Feature-Reduktion ohne echten Performanceverlust

---

## Phase 3.2: Algorithmenvergleich

### Experimentelles Design

Mit fixiertem Setup (CHM-Strategie + selektierte Features) vergleichen wir:

| Kategorie | Algorithmen            | Coarse Grid Configs         |
| --------- | ---------------------- | --------------------------- |
| ML        | Random Forest, XGBoost | 24-48 pro Algorithmus       |
| NN        | 1D-CNN, TabNet         | Baseline + wenige Varianten |

### Coarse Grid Search Strategie

**Warum Coarse statt Fine Search?**

```
Fine Search (verworfen):           Coarse Search (gewählt):
─────────────────────────          ──────────────────────────
• 100+ Configs pro Algo            • 24-48 Configs pro Algo
• Nur 3-Fold CV möglich            • 3-Fold CV ausreichend
• Zu zeitaufwendig                 • Identifiziert beste Region
• Overfitting auf Val-Set          • Fine-Tuning in Phase 3.3
```

**Beispiel Random Forest Coarse Grid:**

```python
{
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
# 3 × 3 × 2 × 2 = 36 Kombinationen → reduziert auf 24 wichtigste
```

### Champion-Selektion

#### Selektionskriterien

1. **Minimum F1 ≥ 0.50:** Filterung nicht-funktionierender Konfigurationen
2. **Train-Val Gap < 35%:** Ausschluss stark overfittender Modelle
3. **Innerhalb Kategorie:** Bestes ML und bestes NN separat wählen

#### Warum zwei Champions (ML + NN)?

```
Szenario: Nur bestes Modell insgesamt
────────────────────────────────────
• Wenn XGBoost gewinnt → Kein NN-Transfer-Vergleich
• Verpassen wir: "Transferieren NNs besser als ML?"

Szenario: Bestes pro Kategorie (gewählt)
────────────────────────────────────────
• ML Champion (RF oder XGBoost)
• NN Champion (1D-CNN oder TabNet)
• Ermöglicht: ML vs. NN Transfer-Vergleich
```

---

## Phase 3.3: Hyperparameter-Optimierung

### Optuna-Konfiguration

| Parameter    | Wert         | Begründung                                                                 |
| ------------ | ------------ | -------------------------------------------------------------------------- |
| **Sampler**  | TPE          | Effizienter als Random bei kontinuierlichen Räumen, nutzt bisherige Trials |
| **Pruner**   | MedianPruner | Bricht Trials ab, die unter Median der bisherigen liegen                   |
| **n_trials** | 50+          | Ausreichend für Konvergenz bei ~5-8 Hyperparametern                        |
| **timeout**  | 2h           | Colab-Runtime-Limit                                                        |
| **CV**       | 3-Fold       | Kompromiss Genauigkeit/Zeit                                                |

### Optuna Search Space (Beispiel XGBoost)

```python
{
    "n_estimators": {"type": "int", "low": 100, "high": 500},
    "max_depth": {"type": "int", "low": 3, "high": 12},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    "min_child_weight": {"type": "int", "low": 1, "high": 10},
    "gamma": {"type": "float", "low": 0, "high": 5}
}
```

### Final Training

Nach HP-Tuning:

```
1. Beste Hyperparameter aus Optuna extrahieren
2. Modell auf Train + Validation trainieren (mehr Daten)
3. Auf Hold-Out Test evaluieren (EINMALIG)
4. Modell speichern für Transfer-Phase
```

**Warum Train + Val für Final Model?**

- Validation war nur für HP-Selektion nötig
- Mehr Trainingsdaten → besseres Modell
- Test bleibt unberührt bis finale Evaluation

---

## Outputs der Berlin-Optimierung

### Metadaten-Dateien

| Datei                     | Inhalt                                                 |
| ------------------------- | ------------------------------------------------------ |
| setup_decisions.json      | CHM-Strategie, selektierte Features, Begründungen      |
| algorithm_comparison.json | Ergebnisse aller Algorithmen, Champion-Auswahl         |
| hp_tuning_ml.json         | Optuna-Trials, beste Parameter für ML                  |
| hp_tuning_nn.json         | Optuna-Trials, beste Parameter für NN                  |
| berlin_evaluation.json    | Test-Metriken, Konfidenzintervalle, Feature Importance |

### Modelle

| Datei                  | Format  | Inhalt                                     |
| ---------------------- | ------- | ------------------------------------------ |
| berlin_ml_champion.pkl | Pickle  | Trainiertes XGBoost/RF mit optimalen HP    |
| berlin_nn_champion.pt  | PyTorch | Trainiertes 1D-CNN/TabNet mit optimalen HP |
| scaler.pkl             | Pickle  | StandardScaler (für Test und Transfer)     |
| label_encoder.pkl      | Pickle  | LabelEncoder (für Genus-Mapping)           |

### Visualisierungen

| Abbildung                       | Zweck                   |
| ------------------------------- | ----------------------- |
| chm_ablation_results.png        | CHM-Varianten-Vergleich |
| pareto_curve.png                | F1 vs. Feature-Anzahl   |
| algorithm_comparison.png        | Alle 4 Algorithmen      |
| optuna_optimization_history.png | HP-Tuning Konvergenz    |
| berlin_confusion_matrix.png     | Finale Test-Performance |
| feature_importance_top20.png    | Wichtigste Features     |

---

## Erwartete Ergebnisse

### Berlin Upper Bound

| Metrik  | Minimum | Target | Begründung                                    |
| ------- | ------- | ------ | --------------------------------------------- |
| Val F1  | 0.55    | 0.60   | Basierend auf Literatur zu Baumklassifikation |
| Test F1 | 0.53    | 0.58   | Leicht niedriger als Val (keine HP-Leak)      |
| Gap     | <35%    | <25%   | Akzeptables Overfitting-Level                 |

### Typische F1-Werte in der Literatur

| Studie                 | Klassen     | Daten         | F1/Accuracy |
| ---------------------- | ----------- | ------------- | ----------- |
| Schiefer et al. (2020) | 7 Arten     | Hyperspektral | ~0.75       |
| Hartling et al. (2019) | 5 Gattungen | S2 + LiDAR    | ~0.65       |
| Immitzer et al. (2016) | 10 Arten    | S2            | ~0.55       |

Unsere Target von 0.60 ist realistisch für 10 Gattungen mit S2 + CHM.

---

_Letzte Aktualisierung: 2026-02-03_
