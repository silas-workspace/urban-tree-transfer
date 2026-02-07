# PRD 003b: Berlin-Optimierung

**PRD ID:** 003b
**Status:** Draft
**Created:** 2026-02-07
**Dependencies:** PRD 003a (setup_decisions.json)

---

## 1. Overview

### 1.1 Problem Statement

With the experimental setup fixed (CHM strategy, feature set, data quality filters), Phase 3b focuses on optimizing classification models for Berlin. This phase selects the best-performing algorithms through systematic comparison, optimizes their hyperparameters, and establishes the Berlin upper bound performance baseline.

Berlin optimization must precede transfer evaluation (003c) and fine-tuning (003d) to ensure we are transferring the best possible models to Leipzig.

### 1.2 Research Questions

| ID      | Question                                                                          | Addressed By    |
| ------- | --------------------------------------------------------------------------------- | --------------- |
| **RQ1** | What is the best achievable performance on Berlin with Sentinel-2 + CHM features? | exp_10, 03b     |
| **RQ1a** | How do different algorithm families (tree-based vs. neural) compare?            | exp_10          |
| **RQ1b** | Do models substantially outperform naive baselines?                              | exp_10, 03b     |
| **RQ1c** | What are the main sources of classification errors?                              | 03b analysis    |

### 1.3 Goals

1. **Algorithm Selection:** Compare 4 algorithms + 3 naive baselines; select 2 champions (1 ML, 1 NN)
2. **Hyperparameter Optimization:** Use Optuna to find optimal HP for both champions
3. **Berlin Upper Bound:** Train final champions on Train+Val, evaluate on hold-out Test
4. **Baseline Validation:** Demonstrate meaningful improvement over naive baselines
5. **Error Analysis:** Comprehensive post-training analysis to understand failure modes

### 1.4 Non-Goals

- Transfer evaluation (covered in PRD 003c)
- Fine-tuning experiments (covered in PRD 003d)
- Multi-seed training (using single seed + bootstrap CIs)
- Ensemble methods (beyond scope)
- Domain adaptation techniques (future work)

---

## 2. Experiments

### 2.1 exp_10_algorithm_comparison.ipynb

**Type:** Exploratory
**Purpose:** Compare all algorithms with coarse HP to select champions

**Inputs:**

- Processed datasets from `03a_setup_fixation.ipynb`
- `outputs/phase_3/metadata/setup_decisions.json`

**Key Tasks:**

1. **Prepare Data**
   - Load datasets with fixed feature set
   - Apply StandardScaler (fit on train, transform val/test)
   - Encode labels with LabelEncoder

2. **Naive Baselines (Imp 2)**
   - **Majority Class Classifier:** Always predict most frequent genus
   - **Stratified Random Classifier:** Random predictions weighted by class distribution
   - **Spatial-Only Random Forest:** Only x/y coordinates (no Sentinel-2, no CHM)
   - Purpose: Establish performance lower bound and validate that spectral features add value

3. **Coarse Grid Search**
   - **Random Forest:** 24 configs (max_depth × min_samples_leaf × min_samples_split)
   - **XGBoost:** 48 configs (n_estimators × max_depth × learning_rate × regularization)
   - **1D-CNN:** Baseline configuration only (HP tuning in 03b)
   - **TabNet:** Baseline configuration only (HP tuning in 03b)
   - Use 3-Fold Spatial Block CV for all evaluations

4. **Collect Metrics**
   - For each algorithm + baselines: Val F1, Train F1, Train-Val Gap, Fit Time
   - Per-genus F1 for error analysis
   - **Relative improvement over best baseline** (in percentage points)

5. **Champion Selection**
   - Apply filters:
     - Minimum Val F1 ≥ 0.50
     - Train-Val Gap < 35%
   - Select:
     - **ML Champion:** Best of (Random Forest, XGBoost)
     - **NN Champion:** Best of (1D-CNN, TabNet)

6. **Visualization**
   - Algorithm comparison bar chart (Val F1 with error bars)
   - Train-Val gap comparison
   - Confusion matrices for top performers
   - **Performance ladder:** Baselines → Default models → Champions (sorted by F1)

**Outputs:**

- `outputs/phase_3/metadata/algorithm_comparison.json`
- `outputs/phase_3/figures/exp_10_algorithm_comparison/`:
  - `algorithm_comparison.png`
  - `confusion_matrix_{champion}.png`
  - `algorithm_train_val_gap.png`
  - `performance_ladder.png` (Imp 2)

**Decision Logic:**

```python
# Filter viable candidates
viable = [
    alg for alg in algorithms
    if alg.val_f1 >= 0.50 and alg.train_val_gap < 0.35
]

# Select champions
ml_champion = max(
    [a for a in viable if a.type == "ml"],
    key=lambda x: x.val_f1
)
nn_champion = max(
    [a for a in viable if a.type == "nn"],
    key=lambda x: x.val_f1
)
```

---

### 2.2 03a_setup_fixation.ipynb (Runner)

**Type:** Runner
**Purpose:** Apply setup decisions and prepare datasets for experiments

**Note:** This notebook is shared with PRD 003a but included here for completeness since it's a prerequisite for exp_10.

**Inputs:**

- `data/phase_2_splits/berlin_*.parquet`
- `data/phase_2_splits/leipzig_*.parquet`
- `outputs/phase_3/metadata/setup_decisions.json`

**Processing Steps:**

1. **Validate Setup Decisions**
   - Load and validate against schema
   - Log CHM strategy, proximity strategy, outlier strategy, and feature count

2. **Apply Dataset Selection**
   - Choose baseline or filtered datasets based on `proximity_strategy`
   - Apply outlier removal based on `outlier_strategy`

3. **Apply Feature Selection**
   - Load selected features from `setup_decisions.json`
   - Apply CHM feature selection
   - Filter columns in all datasets
   - Validate schema consistency

4. **Save Processed Datasets**
   - Save to `data/phase_3_experiments/`
   - Maintain train/val/test/finetune splits

5. **Generate Summary**
   - Feature count, sample counts per city/split
   - Save execution log

**Outputs:**

- `data/phase_3_experiments/berlin_train.parquet`
- `data/phase_3_experiments/berlin_val.parquet`
- `data/phase_3_experiments/berlin_test.parquet`
- `data/phase_3_experiments/leipzig_finetune.parquet`
- `data/phase_3_experiments/leipzig_test.parquet`
- `outputs/phase_3/logs/03a_setup_fixation.json`

---

### 2.3 03b_berlin_optimization.ipynb (Runner)

**Type:** Runner
**Purpose:** HP-tune champions and establish Berlin upper bound

**Inputs:**

- `data/phase_3_experiments/berlin_*.parquet`
- `outputs/phase_3/metadata/algorithm_comparison.json`

**Processing Steps:**

#### Step 1: Load Champions

- Get ML and NN champion names from `algorithm_comparison.json`
- Load corresponding Optuna search spaces from config

#### Step 2: HP-Tuning with Optuna (ML Champion)

- Create Optuna study with TPE sampler
- Run 50+ trials with 3-Fold Spatial Block CV
- Use Hyperband pruning for efficiency
- Objective: Maximize weighted F1
- Track: F1, Train-Val Gap, Trial parameters

#### Step 3: HP-Tuning with Optuna (NN Champion)

- Same procedure as ML champion
- Include early stopping in objective function
- Track validation loss and F1

#### Step 4: Final Training

- Train both champions on **Train + Val** combined with best HP
- Use full training set to maximize performance
- Save trained models with metadata

#### Step 5: Berlin Test Evaluation

**Core Metrics (with Bootstrap CIs):**

- Evaluate on hold-out Berlin Test set
- Compute all metrics with **bootstrap confidence intervals** (1000 resamples, 95% CI):
  - Weighted F1 (primary metric)
  - Macro F1
  - Accuracy
  - Per-genus F1 (with CIs)

**Naive Baseline Comparison (Imp 2):**

- Evaluate 3 naive baselines on same test set:
  - Majority Class
  - Stratified Random
  - Spatial-Only RF
- Compare champion performance against baselines with error bars

**Feature Importance:**

- For ML champion: Gain-based + Permutation importance
- For NN champion: Gradient-based attribution (if applicable)
- Identify top-20 most important features

#### Step 6: Post-Training Error Analysis

**All visualizations use German genus names (`genus_german`).**

**a. Confusion Matrix & Worst Pairs**

- Normalized confusion matrix with German labels
- Extract top-10 most confused genus pairs (off-diagonal entries)
- Publication-quality per-genus metrics table (Precision, Recall, F1, Support)

**b. Conifer vs. Deciduous Analysis**

- Aggregate F1: Nadelbäume vs. Laubbäume
- Hypothesis: Nadelbäume have more distinctive spectral profile → higher F1

**c. Straßen- vs. Anlagenbäume (Berlin only)**

- Per-genus F1 split by `tree_type` (street vs. park trees)
- Hypothesis: Street trees have more homogeneous surroundings → easier to classify?
- Or: Park trees stand more isolated → clearer spectral signal?

**d. Plant Year Impact**

- Bin `plant_year` into decades:
  - Pre-1960
  - 1960-79
  - 1980-99
  - 2000-19
  - 2020+
- Prediction accuracy per decade
- Hypothesis: Younger trees (smaller crown coverage) → harder to classify

**e. Species Breakdown for Problematic Genera**

- For genera with F1 < 0.50: break down by `species_latin`
- Are specific species causing misclassifications?
- Example: If Acer has low F1, is it *Acer platanoides* vs. *Acer pseudoplatanus*?

**f. CHM Value vs. Accuracy**

- Bin CHM_1m values into quantiles
- Compute accuracy per bin
- Hypothesis: Extreme CHM values (very small/tall trees) are harder to classify

**g. Spatial Error Map**

- Join predictions with `geometry_lookup.parquet`
- Compute per-block accuracy (using `block_id` from Phase 2)
- Map: Which parts of Berlin are hardest to classify?

**h. Misclassification Flow**

- Sankey/alluvial diagram: True genus → Predicted genus (errors only)
- Shows systematic error patterns for presentation

#### Step 7: Visualization

**Training & Tuning:**

- HP tuning progress (optimization history)
- Feature importance top-20

**Performance:**

- Performance ladder: Baselines → Champions (with 95% CI error bars)
- Confusion matrix (German labels)
- Per-genus F1 bar chart (German labels)
- Per-genus metrics table

**Error Analysis:**

- Top-10 confused genus pairs
- Conifer vs. Deciduous comparison
- Tree type comparison (Straßen- vs. Anlagenbäume)
- Plant year impact
- Species breakdown for problematic genera
- Spatial error map
- CHM value vs. accuracy
- Misclassification flow diagram

**Total:** ~16 figures for Berlin optimization

**Outputs:**

- **Metadata:**
  - `outputs/phase_3/metadata/hp_tuning_ml.json`
  - `outputs/phase_3/metadata/hp_tuning_nn.json`
  - `outputs/phase_3/metadata/berlin_evaluation.json` (with bootstrap CIs and baseline results)
- **Models:**
  - `outputs/phase_3/models/berlin_ml_champion.pkl`
  - `outputs/phase_3/models/berlin_nn_champion.pt`
  - `outputs/phase_3/models/scaler.pkl` (StandardScaler fitted on Train+Val)
  - `outputs/phase_3/models/label_encoder.pkl` (LabelEncoder)
- **Figures:**
  - `outputs/phase_3/figures/berlin_optimization/*.png` (16 figures)
- **Logs:**
  - `outputs/phase_3/logs/03b_berlin_optimization.json`

---

## 3. Data Flow

```
setup_decisions.json (from PRD 003a)
         │
         ▼
┌─────────────────────────────────────┐
│ 03a_setup_fixation.ipynb (Runner)  │
│ - Apply CHM, proximity, outlier     │
│ - Select features                   │
│ - Save processed datasets           │
└─────────────────────────────────────┘
         │
         ▼
  Processed datasets (data/phase_3_experiments/)
         │
         ▼
┌─────────────────────────────────────┐
│ exp_10_algorithm_comparison.ipynb   │
│ - Naive baselines                   │
│ - Coarse grid search                │
│ - Champion selection (ML + NN)      │
└─────────────────────────────────────┘
         │
         ▼
  algorithm_comparison.json
         │
         ▼
┌─────────────────────────────────────┐
│ 03b_berlin_optimization.ipynb       │
│ - HP tuning (Optuna)                │
│ - Final training (Train+Val)        │
│ - Test evaluation (with CIs)        │
│ - Error analysis (8 analyses)       │
└─────────────────────────────────────┘
         │
         ▼
  Trained Champions + Evaluation Results
         │
         ├──> PRD 003c (Transfer Evaluation)
         └──> PRD 003d (Fine-Tuning)
```

---

## 4. Configuration

### 4.1 Algorithm Configurations

From `configs/experiments/phase3_config.yaml`:

```yaml
# =============================================================================
# PHASE 3.2: ALGORITHM COMPARISON
# =============================================================================
algorithms:
  # ML Algorithms
  random_forest:
    class: sklearn.ensemble.RandomForestClassifier
    type: ml
    coarse_grid:
      n_estimators: [200] # Fixed for speed
      max_depth: [10, 15, 20, null]
      min_samples_leaf: [5, 10, 20]
      min_samples_split: [10, 20]
      class_weight: [balanced]
    optuna_space:
      n_estimators:
        type: int
        low: 100
        high: 500
      max_depth:
        type: int
        low: 5
        high: 30
      min_samples_leaf:
        type: int
        low: 1
        high: 20
      min_samples_split:
        type: int
        low: 2
        high: 20

  xgboost:
    class: xgboost.XGBClassifier
    type: ml
    coarse_grid:
      n_estimators: [200, 300]
      max_depth: [4, 6, 8]
      learning_rate: [0.05, 0.1]
      min_child_weight: [3, 5]
      reg_alpha: [0, 0.1]
      reg_lambda: [1, 2]
      subsample: [0.8]
      colsample_bytree: [0.8]
    optuna_space:
      n_estimators:
        type: int
        low: 100
        high: 500
      max_depth:
        type: int
        low: 3
        high: 10
      learning_rate:
        type: float
        low: 0.01
        high: 0.3
        log: true
      min_child_weight:
        type: int
        low: 1
        high: 10
      reg_alpha:
        type: float
        low: 0
        high: 1
      reg_lambda:
        type: float
        low: 0
        high: 2

  # Neural Networks
  cnn_1d:
    class: urban_tree_transfer.experiments.models.CNN1D
    type: nn
    baseline_config:
      n_temporal_features: 12 # Monthly composites
      n_static_features: null # Computed from data
      n_classes: null # Computed from data
      conv_filters: [64, 128, 128]
      kernel_size: 3
      dropout: 0.3
      dense_units: [256, 128]
      learning_rate: 0.001
      batch_size: 64
      epochs: 50
      early_stopping_patience: 10
    optuna_space:
      conv_filters_depth:
        type: categorical
        choices: [[32, 64], [64, 128, 128], [128, 256, 256]]
      kernel_size:
        type: int
        low: 2
        high: 5
      dropout:
        type: float
        low: 0.1
        high: 0.5
      dense_units:
        type: categorical
        choices: [[128], [256], [256, 128]]
      learning_rate:
        type: float
        low: 0.0001
        high: 0.01
        log: true
      batch_size:
        type: categorical
        choices: [32, 64, 128]

  tabnet:
    class: pytorch_tabnet.tab_model.TabNetClassifier
    type: nn
    baseline_config:
      n_d: 64
      n_a: 64
      n_steps: 5
      gamma: 1.5
      lambda_sparse: 0.001
      optimizer_fn: torch.optim.Adam
      optimizer_params: {lr: 0.02}
      scheduler_params: {step_size: 50, gamma: 0.9}
      mask_type: entmax
      batch_size: 256
      epochs: 100
      patience: 15
    optuna_space:
      n_d:
        type: categorical
        choices: [32, 64, 128]
      n_a:
        type: categorical
        choices: [32, 64, 128]
      n_steps:
        type: int
        low: 3
        high: 7
      gamma:
        type: float
        low: 1.0
        high: 2.0
      lambda_sparse:
        type: float
        low: 0.0001
        high: 0.01
        log: true

# Naive Baselines (Imp 2)
naive_baselines:
  majority_class:
    description: "Always predict most frequent genus"
  stratified_random:
    description: "Random predictions weighted by class distribution"
  spatial_only_rf:
    description: "Random Forest using only x/y coordinates"
    params:
      n_estimators: 100
      max_depth: 10
      class_weight: balanced
```

### 4.2 HP Tuning Settings

```yaml
# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================
hp_tuning:
  optuna:
    n_trials: 50
    timeout_hours: 3.0
    sampler: TPESampler
    pruner: HyperbandPruner
    study_name: "{algorithm}_berlin"
    direction: maximize
    metric: weighted_f1
    cv_folds: 3 # Spatial Block CV
    n_jobs: -1
```

### 4.3 Evaluation Settings

```yaml
# =============================================================================
# EVALUATION METRICS
# =============================================================================
metrics:
  primary: weighted_f1
  secondary:
    - macro_f1
    - accuracy
  per_class: true
  confidence_intervals: true
  ci_method: bootstrap # bootstrap or cross_validation
  n_bootstrap: 1000
  ci_level: 0.95

# Genus grouping for aggregate analyses
genus_groups:
  conifer: [PINUS, PICEA] # Nadelbäume
  deciduous: [TILIA, ACER, QUERCUS, PLATANUS, AESCULUS, BETULA, FRAXINUS, ROBINIA] # Laubbäume

# Plant year bins for post-hoc analysis
plant_year_decades:
  - label: "vor 1960"
    max_year: 1959
  - label: "1960-79"
    min_year: 1960
    max_year: 1979
  - label: "1980-99"
    min_year: 1980
    max_year: 1999
  - label: "2000-19"
    min_year: 2000
    max_year: 2019
  - label: "ab 2020"
    min_year: 2020

# Visualization conventions
visualization:
  use_german_names: true # Always use genus_german for labels
  dpi: 300
  figure_format: png
  figsize_single: [8, 6]
  figsize_double: [14, 6]
  figsize_table: [10, 4]
```

---

## 5. JSON Schemas

### 5.1 Algorithm Comparison Schema

**File:** `schemas/algorithm_comparison.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Algorithm Comparison",
  "type": "object",
  "required": ["timestamp", "algorithms", "baselines", "ml_champion", "nn_champion"],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "setup_reference": { "type": "string" },
    "baselines": {
      "type": "object",
      "description": "Naive baseline results (Imp 2)",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "val_f1_mean": { "type": "number" },
          "val_f1_std": { "type": "number" },
          "description": { "type": "string" }
        }
      }
    },
    "algorithms": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "type": { "type": "string", "enum": ["ml", "nn"] },
          "best_params": { "type": "object" },
          "val_f1_mean": { "type": "number" },
          "val_f1_std": { "type": "number" },
          "train_f1_mean": { "type": "number" },
          "train_val_gap": { "type": "number" },
          "fit_time_seconds": { "type": "number" },
          "passed_filters": { "type": "boolean" },
          "improvement_over_baseline_pp": {
            "type": "number",
            "description": "Improvement over best baseline in percentage points"
          }
        }
      }
    },
    "ml_champion": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "reasoning": { "type": "string" }
      }
    },
    "nn_champion": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "reasoning": { "type": "string" }
      }
    }
  }
}
```

### 5.2 HP Tuning Result Schema

**File:** `schemas/hp_tuning_result.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Hyperparameter Tuning Result",
  "type": "object",
  "required": ["timestamp", "algorithm", "best_params", "best_score"],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "algorithm": { "type": "string" },
    "study_name": { "type": "string" },
    "n_trials": { "type": "integer" },
    "best_trial": { "type": "integer" },
    "best_params": {
      "type": "object",
      "description": "Best hyperparameters found"
    },
    "best_score": {
      "type": "object",
      "properties": {
        "val_f1_mean": { "type": "number" },
        "val_f1_std": { "type": "number" },
        "train_val_gap": { "type": "number" }
      }
    },
    "optimization_history": {
      "type": "array",
      "description": "F1 score per trial for visualization",
      "items": {
        "type": "object",
        "properties": {
          "trial": { "type": "integer" },
          "value": { "type": "number" },
          "params": { "type": "object" }
        }
      }
    },
    "duration_seconds": { "type": "number" }
  }
}
```

### 5.3 Evaluation Metrics Schema

**File:** `schemas/evaluation_metrics.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Evaluation Metrics",
  "type": "object",
  "required": ["timestamp", "model", "dataset", "metrics"],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "model": { "type": "string" },
    "model_path": { "type": "string" },
    "dataset": { "type": "string" },
    "n_samples": { "type": "integer" },
    "metrics": {
      "type": "object",
      "properties": {
        "weighted_f1": { "type": "number" },
        "weighted_f1_ci_lower": { "type": "number" },
        "weighted_f1_ci_upper": { "type": "number" },
        "macro_f1": { "type": "number" },
        "macro_f1_ci_lower": { "type": "number" },
        "macro_f1_ci_upper": { "type": "number" },
        "accuracy": { "type": "number" },
        "accuracy_ci_lower": { "type": "number" },
        "accuracy_ci_upper": { "type": "number" },
        "per_genus_f1": {
          "type": "object",
          "description": "Per-genus F1 scores (latin name → score)",
          "additionalProperties": { "type": "number" }
        },
        "per_genus_f1_ci": {
          "type": "object",
          "description": "Per-genus F1 confidence intervals",
          "additionalProperties": {
            "type": "object",
            "properties": {
              "lower": { "type": "number" },
              "upper": { "type": "number" }
            }
          }
        }
      }
    },
    "confusion_matrix": {
      "type": "array",
      "description": "Confusion matrix as 2D array",
      "items": {
        "type": "array",
        "items": { "type": "integer" }
      }
    },
    "class_labels": {
      "type": "array",
      "description": "Ordered list of class labels (latin names)",
      "items": { "type": "string" }
    },
    "feature_importance": {
      "type": "object",
      "properties": {
        "method": { "type": "string", "enum": ["gain", "permutation", "gradient"] },
        "top_features": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "feature": { "type": "string" },
              "importance": { "type": "number" }
            }
          }
        }
      }
    },
    "baseline_comparison": {
      "type": "object",
      "description": "Comparison with naive baselines (Imp 2)",
      "properties": {
        "majority_f1": { "type": "number" },
        "random_f1": { "type": "number" },
        "spatial_only_f1": { "type": "number" },
        "improvement_over_best_baseline_pp": { "type": "number" }
      }
    },
    "error_analysis": {
      "type": "object",
      "description": "Post-training error analysis results",
      "properties": {
        "worst_confused_pairs": {
          "type": "array",
          "description": "Top-10 most confused genus pairs",
          "items": {
            "type": "object",
            "properties": {
              "true_genus": { "type": "string" },
              "pred_genus": { "type": "string" },
              "count": { "type": "integer" },
              "confusion_rate": { "type": "number" }
            }
          }
        },
        "conifer_vs_deciduous": {
          "type": "object",
          "properties": {
            "conifer_f1": { "type": "number" },
            "deciduous_f1": { "type": "number" }
          }
        },
        "tree_type_comparison": {
          "type": "object",
          "description": "Street vs. park trees (Berlin only)",
          "properties": {
            "street_f1": { "type": "number" },
            "park_f1": { "type": "number" }
          }
        },
        "plant_year_impact": {
          "type": "array",
          "description": "Accuracy by plant year decade",
          "items": {
            "type": "object",
            "properties": {
              "decade": { "type": "string" },
              "accuracy": { "type": "number" },
              "n_samples": { "type": "integer" }
            }
          }
        }
      }
    }
  }
}
```

---

## 6. Visualizations

### 6.1 Algorithm Comparison (exp_10)

| Figure                                 | Filename                      | Description                                         |
| -------------------------------------- | ----------------------------- | --------------------------------------------------- |
| Algorithm F1 comparison                | `algorithm_comparison.png`    | Bar chart: Val F1 for all algorithms + baselines   |
| Champion confusion matrices            | `confusion_matrix_{name}.png` | Normalized confusion matrices (German labels)       |
| Algorithm Train-Val gap                | `algorithm_train_val_gap.png` | Bar chart: Overfitting risk assessment              |
| **Performance ladder (Imp 2)**         | `performance_ladder.png`      | Sorted ranking: Baselines → Models → Champions     |

### 6.2 Berlin Optimization (03b)

**Training & Tuning:**

| Figure                      | Filename                         | Description                                    |
| --------------------------- | -------------------------------- | ---------------------------------------------- |
| Optuna optimization history | `optuna_optimization_history.png` | Trial-by-trial F1 progress                     |
| Feature importance top-20   | `feature_importance_top20.png`    | Bar chart: Top-20 features (gain + permutation) |

**Performance:**

| Figure                        | Filename                             | Description                                  |
| ----------------------------- | ------------------------------------ | -------------------------------------------- |
| Performance ladder (with CIs) | `performance_ladder_with_ci.png`     | Baselines → Champions with 95% CI error bars |
| Confusion matrix              | `berlin_confusion_matrix.png`        | German genus labels                          |
| Per-genus F1 bar chart        | `per_genus_f1_berlin.png`            | German genus labels, sorted by F1            |
| Per-genus metrics table       | `per_genus_metrics_table.png`        | Precision, Recall, F1, Support               |

**Error Analysis:**

| Figure                            | Filename                            | Description                                          |
| --------------------------------- | ----------------------------------- | ---------------------------------------------------- |
| Top-10 confused pairs             | `confusion_pairs_worst.png`         | Most frequent misclassification pairs (German names) |
| Conifer vs. Deciduous             | `conifer_deciduous_comparison.png`  | Aggregate F1: Nadelbäume vs. Laubbäume               |
| Street vs. Park trees             | `tree_type_comparison.png`          | F1 by tree_type (Straßen- vs. Anlagenbäume)          |
| Plant year impact                 | `plant_year_impact.png`             | Accuracy by decade (pre-1960, 1960-79, etc.)         |
| Species breakdown (problematic)   | `species_breakdown_problematic.png` | Species-level detail for low-F1 genera               |
| Spatial error map                 | `spatial_error_map.png`             | Per-block accuracy map of Berlin                     |
| CHM value vs. accuracy            | `chm_impact_on_accuracy.png`        | Accuracy by CHM bin                                  |
| Misclassification flow            | `misclassification_sankey.png`      | Sankey diagram: True → Predicted (errors only)       |

**Total:** ~16 figures

---

## 7. Success Criteria

### 7.1 Functional Requirements

- [ ] `exp_10_algorithm_comparison.ipynb` executes without errors
- [ ] `03a_setup_fixation.ipynb` executes without errors
- [ ] `03b_berlin_optimization.ipynb` executes without errors
- [ ] All JSON outputs validate against schemas
- [ ] All required figures generated (16 total for Berlin optimization)
- [ ] Both champions (ML + NN) saved and loadable

### 7.2 Quality Requirements

**Performance Targets:**

- [ ] Berlin Test Weighted F1 ≥ 0.53 (minimum viable, based on Phase 2 prototyping)
- [ ] Berlin Test Weighted F1 ≥ 0.58 (target)
- [ ] Train-Val Gap < 30% for final champions
- [ ] Champions substantially outperform naive baselines (>15pp improvement)

**Statistical Rigor:**

- [ ] Bootstrap confidence intervals (95%) computed for all key metrics
- [ ] Sample sizes sufficient for stable CIs (n > 1000 for test set)
- [ ] Naive baselines evaluated on same test set for fair comparison

**Error Analysis:**

- [ ] All 8 error analyses completed
- [ ] Top-10 confused pairs identified with German names
- [ ] Per-genus F1 < 0.40 genera investigated at species level

### 7.3 Documentation Requirements

- [ ] Champion selection reasoning documented in `algorithm_comparison.json`
- [ ] HP tuning results logged with optimization history
- [ ] Berlin evaluation metadata includes all error analysis results
- [ ] Execution log saved with timing and dataset info
- [ ] All figures use German genus names for presentation readiness

---

## 8. Dependencies and Outputs

### 8.1 Prerequisites

**From PRD 003a:**

- `outputs/phase_3/metadata/setup_decisions.json` (CHM strategy, feature set, etc.)

**From Phase 2:**

- `data/phase_2_splits/berlin_train.parquet`
- `data/phase_2_splits/berlin_val.parquet`
- `data/phase_2_splits/berlin_test.parquet`
- `data/phase_2_splits/geometry_lookup.parquet` (for spatial error map)

**Configuration:**

- `configs/experiments/phase3_config.yaml` (algorithm configs, HP spaces)

### 8.2 Outputs

**Metadata:**

- `outputs/phase_3/metadata/algorithm_comparison.json` (from exp_10)
- `outputs/phase_3/metadata/hp_tuning_ml.json` (from 03b)
- `outputs/phase_3/metadata/hp_tuning_nn.json` (from 03b)
- `outputs/phase_3/metadata/berlin_evaluation.json` (from 03b, includes CIs and error analysis)

**Models:**

- `outputs/phase_3/models/berlin_ml_champion.pkl`
- `outputs/phase_3/models/berlin_nn_champion.pt`
- `outputs/phase_3/models/scaler.pkl` (StandardScaler)
- `outputs/phase_3/models/label_encoder.pkl` (LabelEncoder)

**Figures:**

- `outputs/phase_3/figures/exp_10_algorithm_comparison/*.png` (4 figures)
- `outputs/phase_3/figures/berlin_optimization/*.png` (16 figures)

**Logs:**

- `outputs/phase_3/logs/03a_setup_fixation.json`
- `outputs/phase_3/logs/03b_berlin_optimization.json`

### 8.3 Consumed By

**PRD 003c (Transfer Evaluation):**

- Trained champions (both models)
- Scaler and label encoder
- Berlin evaluation metrics (for comparison with Leipzig)

**PRD 003d (Fine-Tuning):**

- Trained champions (starting point for fine-tuning)
- Scaler and label encoder
- Berlin evaluation metrics (for baseline comparison)

---

## 9. Source Module Functions

### 9.1 Naive Baselines (Imp 2)

```python
# src/urban_tree_transfer/experiments/models.py

def create_majority_classifier(y_train: np.ndarray) -> Callable:
    """Create majority class baseline that always predicts most frequent class."""

def create_stratified_random_classifier(y_train: np.ndarray) -> Callable:
    """Create stratified random baseline weighted by class distribution."""

def create_spatial_only_rf(
    coords_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 10,
) -> RandomForestClassifier:
    """Create Random Forest using only spatial coordinates (no spectral features)."""
```

### 9.2 HP Tuning

```python
# src/urban_tree_transfer/experiments/hp_tuning.py

def create_optuna_objective(
    model_class: type,
    search_space: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: BaseCrossValidator,
    metric: str = "f1_weighted",
) -> Callable:
    """Create Optuna objective function for HP tuning."""

def run_optuna_study(
    objective: Callable,
    n_trials: int = 50,
    timeout_hours: float = 2.0,
    study_name: str = "hp_tuning",
) -> optuna.Study:
    """Run Optuna study with TPE sampler and Hyperband pruning."""

def extract_best_params(study: optuna.Study) -> dict[str, Any]:
    """Extract best parameters from completed study."""
```

### 9.3 Evaluation with Bootstrap CIs (Imp 4)

```python
# src/urban_tree_transfer/experiments/evaluation.py

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    compute_ci: bool = True,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """
    Comprehensive model evaluation with confidence intervals.

    Returns:
        - metrics: weighted_f1, macro_f1, accuracy (with CIs)
        - per_genus_f1: dict[genus, f1_score] (with CIs)
        - confusion_matrix: 2D array
        - feature_importance: top-20 features (if available)
    """

def compute_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a metric.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """

def evaluate_baselines(
    baselines: dict[str, Callable],
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
) -> dict[str, dict]:
    """Evaluate all naive baselines on test set."""
```

### 9.4 Error Analysis

```python
# src/urban_tree_transfer/experiments/evaluation.py

def analyze_worst_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    german_names: dict[str, str],
    top_k: int = 10,
) -> pd.DataFrame:
    """Extract top-k most confused genus pairs from confusion matrix."""

def analyze_conifer_deciduous(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    genus_groups: dict[str, list[str]],
) -> dict[str, float]:
    """Compute F1 for conifers vs. deciduous trees."""

def analyze_by_metadata(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata_col: pd.Series,
    bins: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Analyze accuracy by metadata column (plant_year, tree_type, CHM bins, etc.).

    If bins provided, apply binning first.
    """

def analyze_spatial_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_ids: pd.Series,
    geometry_lookup: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Compute per-block accuracy and return spatial error map."""

def analyze_species_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genus_labels: list[str],
    species_col: pd.Series,
    f1_threshold: float = 0.50,
) -> dict[str, pd.DataFrame]:
    """
    For genera with F1 < threshold, break down errors by species.

    Returns dict[genus, species_breakdown_df]
    """
```

### 9.5 Visualization

```python
# src/urban_tree_transfer/experiments/visualization.py

def plot_algorithm_comparison(
    results: pd.DataFrame,  # algorithm, val_f1_mean, val_f1_std, type
    baselines: pd.DataFrame,  # baseline, val_f1_mean, val_f1_std
    output_path: Path,
) -> None:
    """Bar chart: Algorithm F1 comparison with baselines."""

def plot_performance_ladder(
    all_results: pd.DataFrame,  # name, val_f1, val_f1_std, category
    output_path: Path,
    highlight_champions: list[str] | None = None,
) -> None:
    """Sorted ranking plot: Baselines → Models → Champions (Imp 2)."""

def plot_optuna_history(
    study: optuna.Study,
    output_path: Path,
) -> None:
    """Optimization history: F1 vs. trial number."""

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],  # German names
    output_path: Path,
    normalize: bool = True,
) -> None:
    """Confusion matrix heatmap with German labels."""

def plot_per_genus_f1(
    f1_scores: dict[str, float],
    german_names: dict[str, str],
    output_path: Path,
    ci_data: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Bar chart: Per-genus F1 (sorted) with optional CIs."""

def plot_confusion_pairs(
    pairs: pd.DataFrame,  # true_genus, pred_genus, count, rate
    german_names: dict[str, str],
    output_path: Path,
) -> None:
    """Bar chart: Top-10 confused pairs."""

def plot_misclassification_sankey(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],  # German names
    output_path: Path,
) -> None:
    """Sankey diagram: True genus → Predicted genus (errors only)."""

def plot_spatial_error_map(
    error_gdf: gpd.GeoDataFrame,  # block_id, geometry, accuracy
    output_path: Path,
    city_name: str = "Berlin",
) -> None:
    """Choropleth map: Per-block accuracy."""

# Additional plotting functions for other error analyses...
```

---

## 10. Design Decisions

### 10.1 Naive Baselines (Imp 2)

**Decision:** Include 3 naive baselines in all evaluations

**Rationale:**

- Establishes performance lower bound
- Validates that spectral features add value over spatial distribution
- Provides context for interpreting champion performance
- Common in ML research (Domingos 2012: "Always compare with simple baselines")

### 10.2 Bootstrap CIs (Imp 4)

**Decision:** Use bootstrap resampling (1000 iterations) for confidence intervals, not multi-seed training

**Rationale:**

- Multi-seed training (e.g., 5 seeds) would 5× training time for marginal benefit
- Bootstrap CIs provide variance estimates on test predictions efficiently
- HP-tuning already explores parameter space stochastically
- Focus computational budget on HP search depth rather than seed repetition

### 10.3 Single Seed

**Decision:** Use fixed seed (42) for all training

**Rationale:**

- Reproducibility: Same results across runs
- Computational efficiency: No need to aggregate multi-seed results
- Optuna trials already explore stochastic variation through HP sampling
- Bootstrap CIs capture prediction uncertainty

### 10.4 Train+Val for Final Training

**Decision:** Train final champions on Train+Val combined, evaluate on Test

**Rationale:**

- Maximizes training data for final model (85% of Berlin data)
- Test set remains completely held-out throughout entire Phase 3
- Standard practice in ML: use validation for HP search, then retrain on Train+Val
- Val set is not needed after HP tuning is complete

### 10.5 German Names for Visualization

**Decision:** All visualizations use `genus_german` from dataset

**Rationale:**

- Präsentation audience is German-speaking
- Improves accessibility and clarity
- Latin names can appear in parentheses where space permits
- No translation needed at presentation time

---

**Status:** Ready for Implementation
