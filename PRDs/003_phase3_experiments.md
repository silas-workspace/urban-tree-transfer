# PRD: Phase 3 Experiments

**PRD ID:** 003
**Status:** Draft
**Created:** 2026-02-03
**Last Updated:** 2026-02-03

---

## 1. Overview

### 1.1 Problem Statement

Phase 2 Feature Engineering has produced ML-ready datasets with spatial train/val/test splits for Berlin and a finetune/test split for Leipzig. Phase 3 must now conduct a systematic series of experiments to answer the core research questions about cross-city transfer learning for tree genus classification.

The experiments must follow a structured methodology:

1. Fix experimental setup (CHM strategy, feature set) before algorithm comparison
2. Optimize models for Berlin (source city) first
3. Evaluate zero-shot transfer to Leipzig (target city)
4. Measure fine-tuning sample efficiency

### 1.2 Research Questions

| ID      | Question                                                                                    | Experiment Phase |
| ------- | ------------------------------------------------------------------------------------------- | ---------------- |
| **RQ1** | What is the best achievable performance on Berlin with Sentinel-2 + CHM features?           | Phase 3.2        |
| **RQ2** | How much does performance drop when applying a Berlin-trained model to Leipzig (zero-shot)? | Phase 3.3        |
| **RQ3** | How much Leipzig fine-tuning data is needed to recover performance?                         | Phase 3.4        |

### 1.3 Goals

1. **Establish Berlin Upper Bound:** Optimal single-city classification performance
2. **Quantify Transfer Gap:** Measure zero-shot performance degradation
3. **Sample Efficiency Curve:** Determine minimum fine-tuning data required
4. **Methodological Rigor:** Reproducible experiments with confidence intervals
5. **Comparable Architectures:** Test both tree-based (ML) and neural (NN) approaches

### 1.4 Non-Goals

- Domain adaptation methods (future work)
- Multi-city joint training (future work)
- Species-level classification
- Real-time inference optimization

---

## 2. Architecture

### 2.1 Directory Structure

```
urban-tree-transfer/
├── configs/
│   ├── cities/                              # Existing from Phase 1
│   ├── features/                            # Existing from Phase 2
│   └── experiments/
│       └── phase3_config.yaml               # NEW: Experiment configuration
├── src/urban_tree_transfer/
│   └── experiments/
│       ├── __init__.py                      # Exports
│       ├── data_loading.py                  # Dataset loading utilities
│       ├── preprocessing.py                 # Feature scaling, encoding
│       ├── models.py                        # Model factory functions
│       ├── training.py                      # Training loops, CV
│       ├── evaluation.py                    # Metrics, confidence intervals
│       ├── transfer.py                      # Transfer-specific metrics
│       ├── ablation.py                      # Ablation study utilities
│       ├── hp_tuning.py                     # Optuna integration
│       └── visualization.py                 # Standardized plots
├── notebooks/
│   ├── runners/
│   │   ├── 03a_setup_fixation.ipynb         # NEW: Fix CHM, features
│   │   ├── 03b_berlin_optimization.ipynb    # NEW: Algorithm comparison, HP-tuning
│   │   ├── 03c_transfer_evaluation.ipynb    # NEW: Zero-shot transfer
│   │   └── 03d_finetuning.ipynb             # NEW: Fine-tuning experiments
│   └── exploratory/
│       ├── exp_07_cross_city_baseline.ipynb # NEW: Descriptive analysis
│       ├── exp_08_chm_ablation.ipynb        # NEW: CHM strategy decision
│       ├── exp_09_feature_reduction.ipynb   # NEW: Top-k feature selection
│       └── exp_10_algorithm_comparison.ipynb # NEW: Coarse HP comparison
├── schemas/
│   ├── setup_decisions.schema.json          # NEW
│   ├── algorithm_comparison.schema.json     # NEW
│   ├── hp_tuning_result.schema.json         # NEW
│   ├── evaluation_metrics.schema.json       # NEW
│   └── finetuning_curve.schema.json         # NEW
├── docs/documentation/
│   └── 03_Experiments/
│       ├── 00_Experiment_Workflow.md        # NEW
│       ├── 01_Setup_Fixation.md             # NEW
│       ├── 02_Berlin_Optimization.md        # NEW
│       ├── 03_Transfer_Evaluation.md        # NEW
│       └── 04_Finetuning.md                 # NEW
└── outputs/
    └── phase_3/                             # NEW
        ├── metadata/
        │   ├── setup_decisions.json         # From exp_08, exp_09
        │   ├── algorithm_comparison.json    # From exp_10
        │   ├── hp_tuning_ml.json            # From 03b
        │   ├── hp_tuning_nn.json            # From 03b
        │   ├── berlin_evaluation.json       # From 03b
        │   ├── transfer_evaluation.json     # From 03c
        │   └── finetuning_curve.json        # From 03d
        ├── models/
        │   ├── berlin_ml_champion.pkl       # Trained ML model
        │   ├── berlin_nn_champion.pt        # Trained NN model
        │   └── finetuned/                   # Fine-tuned checkpoints
        ├── figures/
        │   ├── exp_07_baseline/             # Cross-city visualizations
        │   ├── exp_08_chm_ablation/         # CHM decision plots
        │   ├── exp_09_feature_reduction/    # Feature reduction plots
        │   ├── exp_10_algorithm_comparison/ # Model comparison plots
        │   ├── berlin_optimization/         # HP-tuning, feature importance
        │   ├── transfer/                    # Transfer confusion matrices
        │   └── finetuning/                  # Fine-tuning curves
        └── logs/
            ├── 03a_setup_fixation.json
            ├── 03b_berlin_optimization.json
            ├── 03c_transfer_evaluation.json
            └── 03d_finetuning.json
```

### 2.2 Data Flow

```
Phase 2 Outputs (ML-Ready Datasets, data/phase_2_splits/)
├── berlin_train.parquet (70%)        # ML-optimized (no geometry)
├── berlin_val.parquet (15%)
├── berlin_test.parquet (15%)
├── leipzig_finetune.parquet (80%)
├── leipzig_test.parquet (20%)
├── geometry_lookup.parquet            # tree_id → x/y for visualization
└── *.gpkg                             # Authoritative GeoPackages (kept for traceability)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  exp_07_cross_city_baseline.ipynb (OPTIONAL, parallel)      │
│  ├── Class distribution comparison                          │
│  ├── Phenological profile comparison                        │
│  ├── CHM distribution comparison                            │
│  ├── Feature distribution comparison                        │
│  ├── Cohen's d heatmap (city differences)                   │
│  └── Correlation structure comparison                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼ (figures only, no config output)

┌─────────────────────────────────────────────────────────────┐
│  exp_08_chm_ablation.ipynb                                  │
│  ├── Compare: No CHM vs. zscore vs. percentile vs. both     │
│  ├── Feature importance analysis per variant                │
│  ├── Transfer sanity check (optional)                       │
│  └── Decision logging                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            setup_decisions.json (chm_strategy)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  exp_09_feature_reduction.ipynb                             │
│  ├── RF feature importance ranking                          │
│  ├── Compare: Top-30 vs. Top-50 vs. Top-80 vs. All          │
│  ├── Pareto curve (F1 vs. feature count)                    │
│  └── Decision logging                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            setup_decisions.json (feature_set, selected_features)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  03a_setup_fixation.ipynb (Runner)                          │
│  ├── Load and validate setup_decisions.json                 │
│  ├── Create feature-reduced datasets                        │
│  └── Save processed datasets for experiments                │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            Processed datasets (with fixed setup)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  exp_10_algorithm_comparison.ipynb                          │
│  ├── Random Forest (coarse HP)                              │
│  ├── XGBoost (coarse HP)                                    │
│  ├── 1D-CNN (baseline config)                               │
│  ├── TabNet (baseline config)                               │
│  ├── 5-Fold Spatial Block CV                                │
│  └── Champion selection (1 ML + 1 NN)                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            algorithm_comparison.json (ml_champion, nn_champion)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  03b_berlin_optimization.ipynb (Runner)                     │
│  ├── Load champions from algorithm_comparison.json          │
│  ├── Optuna HP-tuning for ML champion (50+ trials)          │
│  ├── Optuna HP-tuning for NN champion (50+ trials)          │
│  ├── Final training on Train+Val                            │
│  ├── Evaluation on Berlin Test (hold-out)                   │
│  ├── Feature importance analysis                            │
│  ├── Per-genus performance analysis                         │
│  └── Save trained models                                    │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            hp_tuning_ml.json, hp_tuning_nn.json
            berlin_evaluation.json
            berlin_ml_champion.pkl, berlin_nn_champion.pt
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  03c_transfer_evaluation.ipynb (Runner)                     │
│  ├── Load trained Berlin models                             │
│  ├── Zero-shot evaluation on Leipzig Test                   │
│  ├── Transfer gap calculation                               │
│  ├── Per-genus transfer analysis                            │
│  ├── Confusion matrix comparison                            │
│  └── Transfer robustness ranking                            │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            transfer_evaluation.json
            figures/transfer/
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  03d_finetuning.ipynb (Runner)                              │
│  ├── Load best transfer model (ML or NN based on 03c)       │
│  ├── Fine-tune with 10%, 25%, 50%, 100% Leipzig data        │
│  ├── Evaluate each on Leipzig Test                          │
│  ├── Leipzig from-scratch baseline                          │
│  ├── Sample efficiency curve                                │
│  └── Statistical significance tests                         │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            finetuning_curve.json
            figures/finetuning/
```

---

## 3. Configuration

### 3.1 Experiment Config (`configs/experiments/phase3_config.yaml`)

```yaml
# Phase 3 Experiment Configuration
# All hyperparameters, thresholds, and experiment settings

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
global:
  random_seed: 42
  n_jobs: -1 # Use all cores
  cv_folds: 5
  spatial_block_size_m: 500 # From Phase 2

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

# =============================================================================
# PHASE 3.1: SETUP ABLATION
# =============================================================================
setup_ablation:
  # CHM Strategy Comparison
  chm:
    variants:
      - name: no_chm
        features: []
        description: "Sentinel-2 only (baseline, transfer-safe)"
      - name: zscore_only
        features: [CHM_1m_zscore]
        description: "Z-score normalized CHM"
      - name: percentile_only
        features: [CHM_1m_percentile]
        description: "Percentile rank CHM"
      - name: both_engineered
        features: [CHM_1m_zscore, CHM_1m_percentile]
        description: "Both engineered CHM features"
      - name: raw_chm
        features: [CHM_1m]
        description: "Raw CHM (known overfitting risk)"

    # Decision criteria
    decision_rules:
      importance_threshold: 0.25 # CHM >25% importance = problematic
      min_improvement: 0.03 # Must improve >3% to justify CHM
      prefer_simpler: true # Prefer no_chm if difference marginal

  # Feature Reduction
  feature_reduction:
    method: rf_importance # Random Forest feature importance
    variants:
      - name: top_30
        n_features: 30
      - name: top_50
        n_features: 50
      - name: top_80
        n_features: 80
      - name: all_features
        n_features: null # Use all

    # Decision criteria
    decision_rules:
      max_performance_drop: 0.01 # Accept smallest k with ≤1% drop
      prefer_fewer: true

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
        low: 0.0
        high: 1.0
      reg_lambda:
        type: float
        low: 0.0
        high: 5.0
      subsample:
        type: float
        low: 0.6
        high: 1.0
      colsample_bytree:
        type: float
        low: 0.6
        high: 1.0

  # Neural Network Algorithms
  cnn_1d:
    class: urban_tree_transfer.experiments.models.TemporalCNN
    type: nn
    baseline_config:
      n_conv_blocks: 2
      filters: 64
      kernel_size: 3
      dropout: 0.3
      dense_units: 128
      learning_rate: 0.001
      batch_size: 128
      epochs: 100
      early_stopping_patience: 10
    optuna_space:
      n_conv_blocks:
        type: int
        low: 1
        high: 4
      filters:
        type: categorical
        choices: [32, 64, 128]
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
        choices: [64, 128, 256]
      learning_rate:
        type: float
        low: 0.0001
        high: 0.01
        log: true
      batch_size:
        type: categorical
        choices: [64, 128, 256]

  tabnet:
    class: pytorch_tabnet.tab_model.TabNetClassifier
    type: nn
    baseline_config:
      n_d: 32
      n_a: 32
      n_steps: 5
      gamma: 1.3
      lambda_sparse: 0.001
      learning_rate: 0.02
      batch_size: 512
      epochs: 200
      patience: 15
    optuna_space:
      n_d:
        type: categorical
        choices: [8, 16, 32, 64]
      n_a:
        type: categorical
        choices: [8, 16, 32, 64]
      n_steps:
        type: int
        low: 3
        high: 10
      gamma:
        type: float
        low: 1.0
        high: 2.0
      lambda_sparse:
        type: float
        low: 0.0001
        high: 0.01
        log: true

# Champion Selection
champion_selection:
  # Filter criteria (must pass all)
  filters:
    min_val_f1: 0.50 # Minimum validation F1
    max_train_val_gap: 0.35 # Maximum overfitting gap

  # Selection criteria (within passing candidates)
  primary: val_f1 # Maximize validation F1
  tiebreaker: train_val_gap # Minimize gap if tied

# =============================================================================
# PHASE 3.3: HP TUNING
# =============================================================================
hp_tuning:
  method: optuna
  n_trials: 50 # Minimum trials per algorithm
  timeout_hours: 2 # Maximum time per algorithm
  pruning: true # Enable Optuna pruning
  sampler: TPESampler
  cv_folds: 3 # Reduced for speed during tuning

  # Goals
  targets:
    min_val_f1: 0.60
    max_gap: 0.25

# =============================================================================
# PHASE 3.4: TRANSFER EVALUATION
# =============================================================================
transfer:
  # Transfer metrics
  metrics:
    - absolute_drop # F1_source - F1_target
    - relative_drop_pct # (F1_source - F1_target) / F1_source * 100
    - per_genus_transfer # Per-genus F1 comparison

  # Robustness classification thresholds
  robustness_thresholds:
    robust: 0.05 # Δ F1 < 0.05 = robust
    medium: 0.15 # Δ F1 0.05-0.15 = medium
    poor: 1.0 # Δ F1 > 0.15 = poor

# =============================================================================
# PHASE 3.5: FINE-TUNING
# =============================================================================
finetuning:
  # Data fractions to test
  fractions: [0.10, 0.25, 0.50, 1.00]

  # Methods per algorithm type
  methods:
    xgboost:
      strategy: continue_training
      params:
        xgb_model: pretrained # Continue from pretrained model
        n_estimators: 100 # Additional trees
    random_forest:
      strategy: warm_start
      params:
        warm_start: true
        n_estimators_additional: 100
    neural_networks:
      strategy: finetune
      params:
        freeze_layers: null # Fine-tune all layers
        learning_rate_factor: 0.1 # Reduce LR by 10x
        epochs: 50

  # Baseline comparison
  from_scratch_baseline: true # Train Leipzig-only model for comparison

  # Statistical tests
  significance_test: mcnemar # McNemar test for paired comparison
  alpha: 0.05

# =============================================================================
# VISUALIZATION
# =============================================================================
visualization:
  style: seaborn-v0_8-whitegrid
  figsize: [12, 7]
  dpi: 300
  save_format: png

  # Color schemes
  colors:
    berlin: "#1f77b4"
    leipzig: "#ff7f0e"
    algorithms:
      random_forest: "#2ca02c"
      xgboost: "#d62728"
      cnn_1d: "#9467bd"
      tabnet: "#8c564b"
```

### 3.2 Config Loader Extension

Add to `src/urban_tree_transfer/config/loader.py`:

```python
def load_experiment_config() -> dict[str, Any]:
    """Load Phase 3 experiment configuration."""
    config_path = CONFIGS_DIR / "experiments" / "phase3_config.yaml"
    return _load_yaml(config_path)

def get_algorithm_config(algorithm_name: str) -> dict[str, Any]:
    """Get configuration for a specific algorithm."""
    config = load_experiment_config()
    return config["algorithms"][algorithm_name]

def get_coarse_grid(algorithm_name: str) -> dict[str, list]:
    """Get coarse hyperparameter grid for an algorithm."""
    algo_config = get_algorithm_config(algorithm_name)
    return algo_config["coarse_grid"]

def get_optuna_space(algorithm_name: str) -> dict[str, dict]:
    """Get Optuna search space for an algorithm."""
    algo_config = get_algorithm_config(algorithm_name)
    return algo_config["optuna_space"]
```

---

## 4. Implementation Tasks

### 4.1 Exploratory Notebook: exp_07_cross_city_baseline.ipynb

**Purpose:** Descriptive analysis of Berlin vs. Leipzig datasets (hypothesis generation)

**Status:** Optional, not in critical path

**Key Analyses:**

| Analysis                | Visualization                                   | Purpose                              |
| ----------------------- | ----------------------------------------------- | ------------------------------------ |
| Class Distribution      | Stacked bar: Genus × City                       | Identify class imbalance differences |
| Phenological Profiles   | Line plots: NDVI/EVI over months (top-5 genera) | Detect seasonal differences          |
| CHM Distributions       | Violin plots: CHM per genus × city              | Quantify structural differences      |
| Feature Distributions   | Ridge plots: Key features                       | Assess distribution overlap          |
| Statistical Differences | Heatmap: Cohen's d per genus × feature          | Quantify effect sizes                |
| Correlation Structure   | Correlation heatmaps (side-by-side)             | Assess structural similarity         |

**Outputs:**

- `outputs/phase_3/figures/exp_07_baseline/*.png`
- No JSON config (purely descriptive)

---

### 4.2 Exploratory Notebook: exp_08_chm_ablation.ipynb

**Purpose:** Determine CHM strategy through systematic ablation

**Inputs:**

- `data/phase_2_splits/berlin_train.parquet`
- `data/phase_2_splits/berlin_val.parquet`

**Key Tasks:**

1. **Prepare Variants**
   - Create 5 feature sets (no_chm, zscore, percentile, both, raw)
   - Ensure identical S2 features across variants

2. **Cross-Validation Comparison**
   - Use Random Forest with default HP (stable baseline)
   - 5-Fold Spatial Block CV
   - Record: F1, Train-Val Gap, Feature Importance

3. **Feature Importance Analysis**
   - For each variant with CHM: compute importance
   - Flag if CHM features > 25% total importance

4. **Transfer Sanity Check (Optional)**
   - Train on Berlin with each CHM variant
   - Test on Leipzig (small sample)
   - If Δ F1 > 15% with CHM but not without → CHM hurts transfer

5. **Decision Logging**
   - Apply decision rules from config
   - Document reasoning

**Decision Logic:**

```python
# Pseudocode
if chm_importance > 0.25:
    decision = "no_chm"  # CHM dominates = overfitting risk
elif best_chm_f1 - no_chm_f1 < 0.03:
    decision = "no_chm"  # Marginal improvement not worth complexity
elif transfer_drop_with_chm > 0.15:
    decision = "no_chm"  # CHM hurts transfer
else:
    decision = best_chm_variant
```

**Outputs:**

- `outputs/phase_3/metadata/setup_decisions.json` (partial: chm_strategy)
- `outputs/phase_3/figures/exp_08_chm_ablation/*.png`

---

### 4.3 Exploratory Notebook: exp_09_feature_reduction.ipynb

**Purpose:** Determine optimal feature count through importance-based selection

**Inputs:**

- `data/phase_2_splits/berlin_train.parquet`
- `data/phase_2_splits/berlin_val.parquet`
- `outputs/phase_3/metadata/setup_decisions.json` (chm_strategy)

**Key Tasks:**

1. **Compute Feature Importance**
   - Train RF with all features (using CHM decision from exp_08)
   - Extract gain-based importance
   - Rank features

2. **Create Feature Subsets**
   - Top-30, Top-50, Top-80, All features

3. **Evaluate Each Subset**
   - 5-Fold Spatial Block CV with RF
   - Record F1, training time

4. **Pareto Analysis**
   - Plot: F1 vs. Feature Count
   - Identify knee point

5. **Decision Logging**
   - Select smallest k with F1 ≥ F1_all - 0.01
   - Save selected feature list

**Outputs:**

- `outputs/phase_3/metadata/setup_decisions.json` (complete: feature_set, selected_features)
- `outputs/phase_3/figures/exp_09_feature_reduction/*.png`

**Source Module Functions:**

```python
# src/urban_tree_transfer/experiments/ablation.py

def compute_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    method: str = "rf_gain",
) -> pd.DataFrame:
    """Compute and rank feature importance."""

def create_feature_subsets(
    importance_df: pd.DataFrame,
    n_features_list: list[int],
) -> dict[str, list[str]]:
    """Create feature subsets based on importance ranking."""

def evaluate_feature_subsets(
    X: np.ndarray,
    y: np.ndarray,
    feature_subsets: dict[str, list[str]],
    cv: BaseCrossValidator,
) -> pd.DataFrame:
    """Evaluate each feature subset with CV."""

def select_optimal_features(
    results_df: pd.DataFrame,
    max_drop: float = 0.01,
) -> tuple[str, list[str]]:
    """Select optimal feature set based on decision rules."""
```

---

### 4.4 Exploratory Notebook: exp_10_algorithm_comparison.ipynb

**Purpose:** Compare all 4 algorithms with coarse HP to select champions

**Inputs:**

- Processed datasets from 03a_setup_fixation.ipynb
- `outputs/phase_3/metadata/setup_decisions.json`

**Key Tasks:**

1. **Prepare Data**
   - Load datasets with fixed feature set
   - Apply StandardScaler (fit on train)
   - Encode labels

2. **Coarse Grid Search (per algorithm)**
   - Random Forest: 24 configs
   - XGBoost: 48 configs
   - 1D-CNN: baseline only
   - TabNet: baseline only
   - Use 3-Fold Spatial Block CV for speed

3. **Collect Metrics**
   - For each algorithm: Val F1, Train F1, Gap, Fit Time
   - Per-genus F1 for error analysis

4. **Champion Selection**
   - Apply filters (min F1 ≥ 0.50, gap < 35%)
   - Select best ML (RF or XGBoost)
   - Select best NN (1D-CNN or TabNet)

5. **Visualization**
   - Algorithm comparison bar chart
   - Confusion matrices for champions

**Outputs:**

- `outputs/phase_3/metadata/algorithm_comparison.json`
- `outputs/phase_3/figures/exp_10_algorithm_comparison/*.png`

---

### 4.5 Runner Notebook: 03a_setup_fixation.ipynb

**Purpose:** Apply setup decisions and prepare datasets for experiments

**Inputs:**

- `data/phase_2_splits/berlin_*.parquet`
- `data/phase_2_splits/leipzig_*.parquet`
- `outputs/phase_3/metadata/setup_decisions.json`

**Processing Steps:**

1. **Validate Setup Decisions**
   - Load and validate against schema
   - Log CHM strategy and feature count

2. **Apply Feature Selection**
   - Load selected features from setup_decisions.json
   - Filter columns in all datasets
   - Validate schema consistency

3. **Save Processed Datasets**
   - Save to `data/phase_3_experiments/`
   - Maintain train/val/test/finetune splits

4. **Generate Summary**
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

### 4.6 Runner Notebook: 03b_berlin_optimization.ipynb

**Purpose:** HP-tune champions and establish Berlin upper bound

**Inputs:**

- `data/phase_3_experiments/berlin_*.parquet`
- `outputs/phase_3/metadata/algorithm_comparison.json`

**Processing Steps:**

1. **Load Champions**
   - Get ML and NN champion from algorithm_comparison.json
   - Load corresponding search spaces from config

2. **HP-Tuning with Optuna (ML Champion)**
   - Create Optuna study
   - Run 50+ trials with 3-Fold CV
   - Use pruning for efficiency
   - Track: F1, Gap, Trial parameters

3. **HP-Tuning with Optuna (NN Champion)**
   - Same procedure for NN
   - Include early stopping in objective

4. **Final Training**
   - Train both champions on Train + Val with best HP
   - Save trained models

5. **Berlin Test Evaluation**
   - Evaluate on hold-out Berlin Test
   - Compute all metrics with confidence intervals
   - Per-genus analysis
   - Feature importance (gain + permutation for ML)

6. **Visualization**
   - HP tuning progress (optimization history)
   - Confusion matrix
   - Feature importance (top-20)
   - Per-genus F1 bar chart

**Outputs:**

- `outputs/phase_3/metadata/hp_tuning_ml.json`
- `outputs/phase_3/metadata/hp_tuning_nn.json`
- `outputs/phase_3/metadata/berlin_evaluation.json`
- `outputs/phase_3/models/berlin_ml_champion.pkl`
- `outputs/phase_3/models/berlin_nn_champion.pt`
- `outputs/phase_3/figures/berlin_optimization/*.png`
- `outputs/phase_3/logs/03b_berlin_optimization.json`

**Source Module Functions:**

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
    """Run Optuna study with pruning."""

def extract_best_params(study: optuna.Study) -> dict[str, Any]:
    """Extract best parameters from completed study."""
```

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
    """Comprehensive model evaluation with confidence intervals."""

def compute_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence intervals for a metric."""
```

---

### 4.7 Runner Notebook: 03c_transfer_evaluation.ipynb

**Purpose:** Evaluate zero-shot transfer to Leipzig

**Inputs:**

- `data/phase_3_experiments/leipzig_test.parquet`
- `outputs/phase_3/models/berlin_ml_champion.pkl`
- `outputs/phase_3/models/berlin_nn_champion.pt`
- `outputs/phase_3/metadata/berlin_evaluation.json`

**Processing Steps:**

1. **Load Models and Data**
   - Load both trained Berlin models
   - Load Leipzig test set
   - Apply same preprocessing (scaler from training)

2. **Zero-Shot Evaluation**
   - Predict on Leipzig test with both models
   - Compute all metrics

3. **Transfer Gap Analysis**
   - Calculate absolute drop: F1_Berlin - F1_Leipzig
   - Calculate relative drop: (F1_Berlin - F1_Leipzig) / F1_Berlin \* 100
   - Compare ML vs. NN transfer performance

4. **Per-Genus Transfer Analysis**
   - F1 comparison per genus (Berlin vs. Leipzig)
   - Classify robustness: robust (<5%), medium (5-15%), poor (>15%)
   - Identify best/worst transferring genera

5. **Confusion Matrix Comparison**
   - Side-by-side: Berlin vs. Leipzig confusion matrices
   - Highlight systematic differences

6. **Select Best Transfer Model**
   - Compare ML and NN transfer performance
   - Select model for fine-tuning experiments

**Outputs:**

- `outputs/phase_3/metadata/transfer_evaluation.json`
- `outputs/phase_3/figures/transfer/*.png`
- `outputs/phase_3/logs/03c_transfer_evaluation.json`

**Source Module Functions:**

```python
# src/urban_tree_transfer/experiments/transfer.py

def compute_transfer_metrics(
    source_metrics: dict[str, Any],
    target_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compute transfer-specific metrics."""

def compute_per_genus_transfer(
    source_per_genus: dict[str, float],
    target_per_genus: dict[str, float],
    thresholds: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Compute per-genus transfer analysis with robustness classification."""

def create_transfer_summary(
    ml_transfer: dict[str, Any],
    nn_transfer: dict[str, Any],
) -> dict[str, Any]:
    """Create summary comparing ML and NN transfer performance."""
```

---

### 4.8 Runner Notebook: 03d_finetuning.ipynb

**Purpose:** Measure fine-tuning sample efficiency

**Inputs:**

- `data/phase_3_experiments/leipzig_finetune.parquet`
- `data/phase_3_experiments/leipzig_test.parquet`
- Best transfer model from 03c (based on transfer_evaluation.json)

**Processing Steps:**

1. **Load Best Transfer Model**
   - Determine better transfer model (ML or NN)
   - Load pretrained model

2. **Create Fine-Tuning Subsets**
   - 10%, 25%, 50%, 100% of Leipzig finetune data
   - Stratified sampling to maintain class proportions

3. **Fine-Tune at Each Level**
   - For each fraction:
     - Apply fine-tuning strategy (continue training / warm start)
     - Evaluate on Leipzig test
     - Record metrics

4. **Leipzig From-Scratch Baseline**
   - Train new model on 100% Leipzig finetune
   - Compare with transfer + 100% fine-tuning

5. **Statistical Significance**
   - McNemar test: Compare predictions at different levels
   - Identify when fine-tuning significantly improves over zero-shot

6. **Sample Efficiency Curve**
   - Plot: F1 vs. Fine-tuning fraction
   - Mark zero-shot baseline and from-scratch baseline
   - Calculate: fraction needed to reach 90% of from-scratch performance

**Outputs:**

- `outputs/phase_3/metadata/finetuning_curve.json`
- `outputs/phase_3/figures/finetuning/*.png`
- `outputs/phase_3/models/finetuned/*.pkl` or `.pt`
- `outputs/phase_3/logs/03d_finetuning.json`

**Source Module Functions:**

```python
# src/urban_tree_transfer/experiments/training.py

def finetune_xgboost(
    pretrained_model,
    X_finetune: np.ndarray,
    y_finetune: np.ndarray,
    n_additional_estimators: int = 100,
) -> XGBClassifier:
    """Fine-tune XGBoost model with additional trees."""

def finetune_neural_network(
    pretrained_model,
    X_finetune: np.ndarray,
    y_finetune: np.ndarray,
    epochs: int = 50,
    lr_factor: float = 0.1,
) -> nn.Module:
    """Fine-tune neural network with reduced learning rate."""

def create_stratified_subsets(
    X: np.ndarray,
    y: np.ndarray,
    fractions: list[float],
    random_seed: int = 42,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Create stratified subsets at different fractions."""
```

---

## 5. JSON Schemas

### 5.1 Setup Decisions Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Setup Decisions",
  "type": "object",
  "required": ["timestamp", "chm_strategy", "feature_set", "selected_features"],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "chm_strategy": {
      "type": "object",
      "properties": {
        "decision": {
          "type": "string",
          "enum": [
            "no_chm",
            "zscore_only",
            "percentile_only",
            "both_engineered",
            "raw_chm"
          ]
        },
        "reasoning": { "type": "string" },
        "ablation_results": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "variant": { "type": "string" },
              "val_f1_mean": { "type": "number" },
              "val_f1_std": { "type": "number" },
              "chm_importance": { "type": ["number", "null"] },
              "train_val_gap": { "type": "number" }
            }
          }
        }
      }
    },
    "feature_set": {
      "type": "object",
      "properties": {
        "decision": { "type": "string" },
        "n_features": { "type": "integer" },
        "reasoning": { "type": "string" },
        "pareto_results": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "variant": { "type": "string" },
              "n_features": { "type": "integer" },
              "val_f1_mean": { "type": "number" },
              "val_f1_std": { "type": "number" }
            }
          }
        }
      }
    },
    "selected_features": {
      "type": "array",
      "items": { "type": "string" }
    }
  }
}
```

### 5.2 Algorithm Comparison Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Algorithm Comparison",
  "type": "object",
  "required": ["timestamp", "algorithms", "ml_champion", "nn_champion"],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "setup_reference": { "type": "string" },
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
          "passed_filters": { "type": "boolean" }
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

### 5.3 Evaluation Metrics Schema

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
        "accuracy": { "type": "number" },
        "per_genus_f1": {
          "type": "object",
          "additionalProperties": { "type": "number" }
        }
      }
    },
    "confusion_matrix": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "integer" }
      }
    },
    "feature_importance": {
      "type": "object",
      "properties": {
        "method": { "type": "string" },
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
    }
  }
}
```

### 5.4 Transfer Evaluation Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Transfer Evaluation",
  "type": "object",
  "required": ["timestamp", "source_city", "target_city", "models"],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "source_city": { "type": "string" },
    "target_city": { "type": "string" },
    "models": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "source_f1": { "type": "number" },
          "target_f1": { "type": "number" },
          "absolute_drop": { "type": "number" },
          "relative_drop_pct": { "type": "number" },
          "per_genus_transfer": {
            "type": "object",
            "additionalProperties": {
              "type": "object",
              "properties": {
                "source_f1": { "type": "number" },
                "target_f1": { "type": "number" },
                "drop": { "type": "number" },
                "robustness": {
                  "type": "string",
                  "enum": ["robust", "medium", "poor"]
                }
              }
            }
          }
        }
      }
    },
    "best_transfer_model": { "type": "string" },
    "selection_reasoning": { "type": "string" }
  }
}
```

### 5.5 Fine-Tuning Curve Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Fine-Tuning Curve",
  "type": "object",
  "required": ["timestamp", "model", "results"],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "model": { "type": "string" },
    "finetuning_method": { "type": "string" },
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "fraction": { "type": "number" },
          "n_samples": { "type": "integer" },
          "target_f1": { "type": "number" },
          "target_f1_ci_lower": { "type": "number" },
          "target_f1_ci_upper": { "type": "number" },
          "improvement_over_zeroshot": { "type": "number" },
          "pct_of_from_scratch": { "type": "number" }
        }
      }
    },
    "baselines": {
      "type": "object",
      "properties": {
        "zero_shot_f1": { "type": "number" },
        "from_scratch_f1": { "type": "number" }
      }
    },
    "efficiency_metrics": {
      "type": "object",
      "properties": {
        "fraction_to_match_scratch": { "type": ["number", "null"] },
        "fraction_to_90pct_scratch": { "type": "number" }
      }
    },
    "significance_tests": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "comparison": { "type": "string" },
          "test": { "type": "string" },
          "p_value": { "type": "number" },
          "significant": { "type": "boolean" }
        }
      }
    }
  }
}
```

---

## 6. Visualization Standards

### 6.1 Required Figures

| Experiment | Figure                        | Filename                                  |
| ---------- | ----------------------------- | ----------------------------------------- |
| exp_07     | Class distribution comparison | `genus_distribution_comparison.png`       |
| exp_07     | Phenological profiles         | `phenological_profiles_top5.png`          |
| exp_07     | CHM violin per genus          | `chm_violin_per_genus.png`                |
| exp_07     | Cohen's d heatmap             | `cohens_d_heatmap.png`                    |
| exp_08     | CHM ablation comparison       | `chm_ablation_results.png`                |
| exp_08     | CHM feature importance        | `chm_feature_importance.png`              |
| exp_09     | Feature importance ranking    | `feature_importance_ranking.png`          |
| exp_09     | Pareto curve                  | `pareto_curve.png`                        |
| exp_10     | Algorithm comparison          | `algorithm_comparison.png`                |
| exp_10     | Champion confusion matrices   | `confusion_matrix_{champion}.png`         |
| 03b        | HP tuning history             | `optuna_optimization_history.png`         |
| 03b        | Berlin confusion matrix       | `berlin_confusion_matrix.png`             |
| 03b        | Feature importance top-20     | `feature_importance_top20.png`            |
| 03b        | Per-genus F1                  | `per_genus_f1_berlin.png`                 |
| 03c        | Transfer comparison           | `transfer_comparison.png`                 |
| 03c        | Confusion comparison          | `confusion_comparison_berlin_leipzig.png` |
| 03c        | Per-genus transfer            | `per_genus_transfer_robustness.png`       |
| 03d        | Fine-tuning curve             | `finetuning_curve.png`                    |
| 03d        | Comparison with baselines     | `finetuning_vs_baselines.png`             |

### 6.2 Visualization Module

```python
# src/urban_tree_transfer/experiments/visualization.py

def setup_plot_style() -> None:
    """Set up publication-ready plot style."""

def plot_algorithm_comparison(
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create algorithm comparison bar chart."""

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    normalize: bool = True,
) -> None:
    """Create confusion matrix heatmap."""

def plot_confusion_comparison(
    cm_source: np.ndarray,
    cm_target: np.ndarray,
    labels: list[str],
    source_name: str,
    target_name: str,
    output_path: Path,
) -> None:
    """Create side-by-side confusion matrix comparison."""

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    output_path: Path,
) -> None:
    """Create feature importance bar chart."""

def plot_finetuning_curve(
    results: list[dict],
    baselines: dict[str, float],
    output_path: Path,
) -> None:
    """Create fine-tuning sample efficiency curve."""

def plot_per_genus_transfer(
    transfer_data: dict[str, dict],
    output_path: Path,
) -> None:
    """Create per-genus transfer robustness visualization."""
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```
tests/experiments/
├── test_data_loading.py
├── test_preprocessing.py
├── test_models.py
├── test_training.py
├── test_evaluation.py
├── test_transfer.py
├── test_ablation.py
├── test_hp_tuning.py
└── test_visualization.py
```

**Key Test Cases:**

| Module        | Test                      | Purpose                                 |
| ------------- | ------------------------- | --------------------------------------- |
| evaluation.py | test_compute_metrics      | Verify metrics match sklearn            |
| evaluation.py | test_confidence_intervals | Verify CI coverage                      |
| transfer.py   | test_transfer_metrics     | Verify gap calculations                 |
| ablation.py   | test_decision_logic       | Verify decision rules applied correctly |
| hp_tuning.py  | test_optuna_objective     | Verify objective returns valid scores   |

### 7.2 Integration Tests

```
tests/integration/
└── test_phase3_pipeline.py
```

**Test Cases:**

- End-to-end: Setup → Algorithm → Berlin → Transfer → Finetune (on small subset)
- Schema validation for all JSON outputs
- Figure generation without errors

---

## 8. Execution Order

### 8.1 Critical Path

```
1. exp_08_chm_ablation.ipynb
   └── Output: setup_decisions.json (partial)

2. exp_09_feature_reduction.ipynb
   └── Output: setup_decisions.json (complete)

3. 03a_setup_fixation.ipynb (Runner)
   └── Output: Processed datasets

4. exp_10_algorithm_comparison.ipynb
   └── Output: algorithm_comparison.json

5. 03b_berlin_optimization.ipynb (Runner)
   └── Output: hp_tuning_*.json, berlin_evaluation.json, models

6. 03c_transfer_evaluation.ipynb (Runner)
   └── Output: transfer_evaluation.json

7. 03d_finetuning.ipynb (Runner)
   └── Output: finetuning_curve.json
```

### 8.2 Optional/Parallel

```
exp_07_cross_city_baseline.ipynb
└── Can run anytime (no dependencies, no outputs used by others)
```

---

## 9. Colab Workflow

### 9.1 Drive Structure

```
Google Drive/
└── dev/urban-tree-transfer/
    ├── data/
    │   ├── phase_2_splits/                    # Input from Phase 2c
    │   │   ├── berlin_train.parquet            # ML-optimized (no geometry)
    │   │   ├── berlin_val.parquet
    │   │   ├── berlin_test.parquet
    │   │   ├── leipzig_finetune.parquet
    │   │   ├── leipzig_test.parquet
    │   │   ├── *_filtered.parquet              # Filtered variants
    │   │   ├── geometry_lookup.parquet         # tree_id → x/y for visualization
    │   │   └── *.gpkg                          # Authoritative GeoPackages (kept)
    │   └── phase_3_experiments/                # Processed for experiments
    │       ├── berlin_train.parquet
    │       ├── berlin_val.parquet
    │       ├── berlin_test.parquet
    │       ├── leipzig_finetune.parquet
    │       └── leipzig_test.parquet
    ├── outputs/
    │   └── phase_3/
    │       ├── metadata/                  # JSON configs
    │       ├── models/                    # Trained models
    │       ├── figures/                   # Visualizations
    │       └── logs/                      # Execution logs
    └── repo/                              # Cloned repository
```

### 9.2 Notebook Header Template

```python
# ============================================================================
# SETUP (Run once per session)
# ============================================================================

# Install package from GitHub
!pip install git+https://github.com/SilasPignotti/urban-tree-transfer.git -q

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define paths
DRIVE_BASE = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
DATA_DIR = DRIVE_BASE / "data"
OUTPUT_DIR = DRIVE_BASE / "outputs/phase_3"

# Create output directories
(OUTPUT_DIR / "metadata").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

# Import project modules
from urban_tree_transfer.config import load_experiment_config
from urban_tree_transfer.experiments import (
    data_loading,
    preprocessing,
    models,
    training,
    evaluation,
    transfer,
    visualization,
)
```

### 9.3 Sync Workflow

1. **Before Running Notebook:**
   - Pull latest code to local repo
   - Push to GitHub
   - Colab installs from GitHub

2. **After Running Notebook:**
   - Download from Drive: `outputs/phase_3/metadata/*.json`
   - Download from Drive: `outputs/phase_3/figures/*.png`
   - Commit to local repo
   - Push to GitHub

3. **Model Handling:**
   - Models stay on Drive (too large for Git)
   - Reference models by Drive path in notebooks

---

## 10. Success Criteria

### 10.1 Functional Requirements

- [ ] All 4 exploratory notebooks execute without errors
- [ ] All 4 runner notebooks execute without errors
- [ ] All JSON outputs validate against schemas
- [ ] All required figures generated
- [ ] Models saved and loadable

### 10.2 Quality Requirements

- [ ] Berlin validation F1 ≥ 0.55 (minimum viable)
- [ ] Berlin validation F1 ≥ 0.60 (target)
- [ ] Train-Val gap < 30%
- [ ] Confidence intervals computed for all key metrics
- [ ] Statistical significance tests for fine-tuning comparisons

### 10.3 Documentation Requirements

- [ ] Methodology documentation complete
- [ ] All decisions logged with reasoning
- [ ] Execution logs for all notebooks

---

## 11. Timeline Estimate

| Task                     | Estimated Hours |
| ------------------------ | --------------- |
| Config + Schemas         | 2h              |
| src/experiments/ modules | 8h              |
| exp_07 (optional)        | 2h              |
| exp_08 + exp_09          | 4h              |
| 03a runner               | 2h              |
| exp_10                   | 4h              |
| 03b runner               | 6h              |
| 03c runner               | 3h              |
| 03d runner               | 4h              |
| Testing                  | 4h              |
| Documentation            | 3h              |
| **Total**                | **~42h**        |

---

## 12. Design Decisions

The following methodological decisions were made during planning:

### 12.1 NN Architecture: 1D-CNN

**Decision:** Use 1D-CNN for temporal modeling (not LSTM or Transformer)

**Rationale:**

- Sequence length is only 12 time points (monthly composites) — too short for LSTM/Transformer benefits
- 1D-CNN captures local temporal patterns efficiently (spring slope, summer peak)
- Fewer parameters → more robust with limited training data (~10k-30k samples)
- Faster training and simpler hyperparameter tuning
- LSTM/Transformer would be appropriate for sequences with 100+ time points

### 12.2 Fine-tuning Strategy: Single Approach

**Decision:** Use single fine-tuning strategy per model type:

- **ML (XGBoost):** Warm-start with additional estimators
- **NN (1D-CNN/TabNet):** Full fine-tune with 0.1× original learning rate

**Rationale:**

- Testing multiple strategies (freeze layers, discriminative LR, gradual unfreezing) would multiply experiment time
- Already testing 4 fine-tuning fractions (10%, 25%, 50%, 100%)
- Strategy comparison is a separate research question beyond scope

### 12.3 Seed Strategy: Single Seed + Bootstrap CIs

**Decision:** Use single seed (42) for training, bootstrap confidence intervals for evaluation

**Rationale:**

- Multi-seed (e.g., 5 seeds) would 5× training time
- Bootstrap CIs (1000 resamples) provide variance estimates on test predictions
- HP-tuning already has inherent randomness across trials
- Optional: Final model with 3 seeds as stretch goal

### 12.4 Class Weighting: Recompute for Leipzig

**Decision:** Recompute class weights based on Leipzig distribution during fine-tuning

**Rationale:**

- Fine-tuning optimizes for Leipzig performance, not Berlin
- Leipzig has different genus distribution than Berlin
- Using `class_weight='balanced'` with Leipzig data ensures model adapts to local class frequencies
- From-scratch Leipzig baseline also uses Leipzig weights for fair comparison

---

**Status:** Ready for Implementation
