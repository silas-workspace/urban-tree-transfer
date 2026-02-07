# PRD 003d: Fine-Tuning

**PRD ID:** 003d
**Status:** Draft
**Created:** 2026-02-07
**Dependencies:** PRD 003b (trained Berlin champions)

---

## 1. Overview

### 1.1 Problem Statement

PRD 003c (Transfer Evaluation) quantifies the zero-shot transfer gap when applying Berlin-trained models to Leipzig. PRD 003d addresses the next critical question: **How much Leipzig data is needed to recover performance through fine-tuning?**

This experiment measures **sample efficiency** - the relationship between fine-tuning data quantity and performance recovery. Understanding this relationship is crucial for practical deployment scenarios where labeled target-city data is expensive to acquire.

### 1.2 Research Question

**RQ3:** How much Leipzig fine-tuning data is needed to recover performance?

Specifically:
- What is the minimum data fraction needed to reach 90% of from-scratch performance?
- Does the relationship follow a predictable power-law curve?
- Do ML and NN architectures differ in sample efficiency?
- Which genera recover fastest with local data?

### 1.3 Goals

1. **Sample Efficiency Curves:** Plot performance vs. fine-tuning data fraction for both champions
2. **From-Scratch Comparison:** Establish Leipzig-only baselines to measure recovery completeness
3. **Power-Law Modeling (Improvement 6):** Fit power-law curves (y = a × x^b) to extrapolate data requirements
4. **Statistical Validation:** Use McNemar tests to identify significant improvement thresholds
5. **Per-Genus Analysis:** Identify which genera benefit most from fine-tuning
6. **Architecture Comparison:** Compare ML vs. NN sample efficiency

### 1.4 Non-Goals

- Algorithm comparison across all architectures (only the two champions from 003b)
- Transfer gap analysis (completed in 003c)
- Domain adaptation methods (future work)
- Multi-stage fine-tuning strategies

---

## 2. Experiment: 03d Fine-Tuning

### 2.1 Purpose

Measure fine-tuning sample efficiency for both Berlin champions (ML and NN) on Leipzig data.

### 2.2 Inputs

- `data/phase_3_experiments/leipzig_finetune.parquet` (Leipzig fine-tuning set)
- `data/phase_3_experiments/leipzig_test.parquet` (Leipzig test set)
- `outputs/phase_3/models/berlin_ml_champion.pkl` (from 003b)
- `outputs/phase_3/models/berlin_nn_champion.pt` (from 003b)
- `outputs/phase_3/models/scaler.pkl` (Berlin scaler from 003b)
- `outputs/phase_3/metadata/transfer_evaluation.json` (zero-shot baselines)

### 2.3 Processing Steps

#### 2.3.1 Load Both Champions

1. Load ML champion (XGBoost or RandomForest) from 003b
2. Load NN champion (CNN-1D or TabNet) from 003b
3. Load Berlin scaler (reused for fine-tuning - models expect Berlin-scaled features)
4. Load zero-shot results from transfer_evaluation.json for baseline comparison

#### 2.3.2 Create Fine-Tuning Subsets

1. Generate stratified subsets at fractions: **[10%, 25%, 50%, 100%]**
2. Use stratified sampling to maintain class (genus) proportions
3. Use same random seed (42) for both models to ensure fair comparison
4. Apply Berlin scaler to all subsets (model expects Berlin-scaled features)

**Implementation:**
```python
def create_stratified_subsets(
    X: np.ndarray,
    y: np.ndarray,
    fractions: list[float],
    random_seed: int = 42,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Create stratified subsets at different fractions."""
```

#### 2.3.3 Fine-Tune Both Models at Each Level

For each fraction × each model:

**ML Fine-Tuning (Warm-Start):**
- XGBoost: Continue training with `xgb_model=pretrained`, add 100 trees
- RandomForest: Use `warm_start=True`, add 100 trees
- Keep all other hyperparameters from 003b
- Apply class weighting if imbalanced

**NN Fine-Tuning (Full Fine-Tune with Reduced LR):**
- Fine-tune all layers (no freezing)
- Reduce learning rate by 10× (lr_factor = 0.1)
- Train for 50 epochs with early stopping
- Use same architecture and hyperparameters from 003b

**Evaluation:**
- Evaluate on Leipzig test set
- Compute full metrics (weighted F1, macro F1, accuracy, per-genus F1)
- Record predictions for McNemar tests

#### 2.3.4 Leipzig From-Scratch Baselines

1. Train ML model from scratch on 100% Leipzig finetune data
2. Train NN model from scratch on 100% Leipzig finetune data
3. Fit **new scaler** on Leipzig finetune data (independent from Berlin)
4. Use same architectures and hyperparameters as 003b champions
5. Compare performance with transfer + 100% fine-tuning

**Purpose:** Determine if transfer learning provides any advantage over city-specific training.

#### 2.3.5 Statistical Significance Testing

Use **McNemar test** for paired prediction comparison:
- Compare each fine-tuning level vs. zero-shot
- Compare each fine-tuning level vs. from-scratch
- Compare ML vs. NN at each level
- Identify minimum fraction for significant improvement (p < 0.05)

#### 2.3.6 Sample Efficiency Curves

**Standard Curve:**
- Plot: F1 vs. Fine-tuning fraction (0%, 10%, 25%, 50%, 100%)
- Mark zero-shot baseline (0% local data)
- Mark from-scratch baseline (100% local data, no transfer)
- Calculate fraction to reach 90% of from-scratch performance

**Improvement 6: Power-Law Fit**
- Fit power-law model: **Performance = a × N^b**
- Use `scipy.optimize.curve_fit` with function `f(N) = a * N**b`
- Compute R² goodness-of-fit
- Extrapolate to calculate 95% recovery point
- Compare exponent `b` with literature (Tong et al. 2019: b ≈ 0.35)

**Interpretation:**
- **b < 0.5:** Diminishing returns - early data very valuable
- **b ≈ 0.5:** Square-root scaling (typical for ML)
- **b > 0.5:** Accelerating returns - later data more valuable

#### 2.3.7 Per-Genus Recovery Analysis

1. Compute per-genus F1 at each fine-tuning fraction
2. Create heatmap: genera (rows) × fractions (columns)
3. Identify fastest-recovering genera
4. Test hypothesis: Do genera with poor zero-shot transfer benefit most from fine-tuning?

#### 2.3.8 ML vs. NN Comparison

1. Side-by-side sample efficiency curves
2. Identify crossover point: At which fraction does fine-tuning surpass from-scratch?
3. Compare recovery rates (slope of curve)
4. McNemar significance heatmap across all comparison pairs

### 2.4 Outputs

**Metadata:**
- `outputs/phase_3/metadata/finetuning_curve.json` (with power_law_fit, recovery_points)

**Figures:**
- `outputs/phase_3/figures/finetuning/finetuning_curve.png` (ML + NN curves with power-law fit)
- `outputs/phase_3/figures/finetuning/finetuning_vs_baselines.png`
- `outputs/phase_3/figures/finetuning/finetuning_per_genus_recovery.png`
- `outputs/phase_3/figures/finetuning/finetuning_ml_vs_nn_comparison.png`
- `outputs/phase_3/figures/finetuning/finetuning_significance_matrix.png`
- `outputs/phase_3/figures/finetuning/finetuning_powerlaw_extrapolation.png` (log-scale, with 95% recovery marker)

**Models:**
- `outputs/phase_3/models/finetuned/ml_champion_f{fraction}.pkl` (4 fractions)
- `outputs/phase_3/models/finetuned/nn_champion_f{fraction}.pt` (4 fractions)
- `outputs/phase_3/models/leipzig_ml_scratch.pkl`
- `outputs/phase_3/models/leipzig_nn_scratch.pt`

**Logs:**
- `outputs/phase_3/logs/03d_finetuning.json`

---

## 3. Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRD 003d: Fine-Tuning                        │
└─────────────────────────────────────────────────────────────────┘

INPUTS (from PRD 003b):
  - berlin_ml_champion.pkl
  - berlin_nn_champion.pt
  - scaler.pkl (Berlin scaler, reused)

INPUTS (from Phase 2):
  - leipzig_finetune.parquet
  - leipzig_test.parquet

PROCESSING:
  1. Create stratified subsets [10%, 25%, 50%, 100%]
  2. Fine-tune both champions at each fraction
  3. Train from-scratch baselines (Leipzig-only, new scaler)
  4. Evaluate all on Leipzig test
  5. Fit power-law curves (Improvement 6)
  6. McNemar significance tests

OUTPUTS:
  - finetuning_curve.json (with power_law_fit, recovery_points)
  - 6 visualization figures
  - 8 fine-tuned models + 2 from-scratch baselines
  - 03d_finetuning.json log

PARALLEL TO: PRD 003c (both use trained champions from 003b)
```

---

## 4. Configuration

Extract from `configs/experiments/phase3_config.yaml`:

```yaml
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

  # Preprocessing
  preprocessing:
    transfer_scaler: berlin # Keep Berlin scaler for fine-tuning (model expects Berlin-scaled features)
    from_scratch_scaler: leipzig # Fit new scaler on Leipzig for from-scratch baseline

  # Baseline comparison
  from_scratch_baseline: true # Train Leipzig-only model for comparison

  # Statistical tests
  significance_test: mcnemar # McNemar test for paired comparison
  alpha: 0.05

  # Power-law modeling (Improvement 6)
  power_law:
    enabled: true
    extrapolate_to: 0.95 # Calculate data needed for 95% recovery
    literature_comparison:
      reference: "Tong et al. (2019)"
      expected_exponent_range: [0.30, 0.40]
      expected_label_savings: [0.50, 0.70] # 50-70% savings reported
```

---

## 5. JSON Schema

### 5.1 Fine-Tuning Curve Schema

File: `src/urban_tree_transfer/schemas/finetuning_curve.schema.json`

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
    },
    "power_law_fit": {
      "type": "object",
      "description": "Power-law model parameters for sample efficiency curve (Improvement 6)",
      "properties": {
        "a": { "type": "number", "description": "Scaling coefficient" },
        "b": { "type": "number", "description": "Exponent (b < 0.5 = diminishing returns)" },
        "r_squared": { "type": "number", "description": "Goodness of fit (0-1)" },
        "formula": { "type": "string", "description": "Human-readable formula" }
      },
      "required": ["a", "b", "r_squared", "formula"]
    },
    "recovery_points": {
      "type": "object",
      "description": "Extrapolated recovery points (Improvement 6)",
      "properties": {
        "n_samples_90pct": { "type": "integer", "description": "Samples needed for 90% recovery" },
        "n_samples_95pct": { "type": "integer", "description": "Samples needed for 95% recovery" },
        "fraction_to_90pct": { "type": "number", "description": "Fraction needed for 90% recovery" },
        "fraction_to_95pct": { "type": "number", "description": "Fraction needed for 95% recovery" },
        "fraction_to_match_scratch": { "type": ["number", "null"], "description": "Fraction to match from-scratch (null if never)" }
      },
      "required": ["n_samples_90pct", "n_samples_95pct", "fraction_to_90pct", "fraction_to_95pct"]
    }
  }
}
```

---

## 6. Visualizations

### 6.1 Required Figures

| Figure | Filename | Description |
|--------|----------|-------------|
| Fine-tuning curve (ML + NN, with power-law fit) | `finetuning_curve.png` | Side-by-side sample efficiency curves with fitted power-law lines |
| Comparison with baselines | `finetuning_vs_baselines.png` | Show zero-shot and from-scratch baselines |
| Per-genus F1 recovery | `finetuning_per_genus_recovery.png` | Heatmap: genera × fractions |
| ML vs. NN comparison | `finetuning_ml_vs_nn_comparison.png` | Direct comparison of sample efficiency |
| McNemar significance matrix | `finetuning_significance_matrix.png` | Heatmap of p-values |
| **Power-law extrapolation (Improvement 6)** | `finetuning_powerlaw_extrapolation.png` | Log-scale plot with 95% recovery marker |

### 6.2 Visualization Functions

From `src/urban_tree_transfer/experiments/visualization.py`:

```python
# --- Fine-Tuning Plots (03d) ---

def plot_finetuning_curve(
    results: list[dict],
    baselines: dict[str, float],
    power_law_params: dict[str, float] | None = None,
    output_path: Path,
) -> None:
    """
    Sample efficiency curve with zero-shot + from-scratch baselines.

    Improvement 6: If power_law_params provided, overlay fitted curve.
    """

def plot_finetuning_ml_vs_nn(
    ml_results: list[dict],
    nn_results: list[dict],
    output_path: Path,
) -> None:
    """Side-by-side ML vs. NN fine-tuning curves."""

def plot_finetuning_per_genus_recovery(
    per_genus_by_fraction: dict[float, dict[str, float]],
    genus_labels: list[str],  # German names
    output_path: Path,
) -> None:
    """Heatmap: per-genus F1 at each fine-tuning fraction."""

def plot_significance_matrix(
    p_values: pd.DataFrame,  # comparison pairs × p-values
    output_path: Path,
) -> None:
    """Heatmap of McNemar test p-values across comparisons."""

def plot_powerlaw_extrapolation(
    observed_points: list[dict],  # {fraction, n_samples, f1}
    power_law_params: dict[str, float],  # {a, b, r_squared}
    recovery_points: dict[str, int],  # {n_samples_90pct, n_samples_95pct}
    output_path: Path,
) -> None:
    """
    Log-scale plot showing:
    - Observed data points
    - Fitted power-law curve
    - Extrapolation to 95% recovery point
    - Literature comparison annotation (Tong et al. 2019)

    Improvement 6 implementation.
    """
```

---

## 7. Fine-Tuning Strategies

### 7.1 ML Fine-Tuning (Warm-Start)

**XGBoost:**
```python
def finetune_xgboost(
    pretrained_model: XGBClassifier,
    X_finetune: np.ndarray,
    y_finetune: np.ndarray,
    n_additional_estimators: int = 100,
) -> XGBClassifier:
    """
    Fine-tune XGBoost model with additional trees.

    Strategy: Continue training using xgb_model parameter.
    This adds new trees while keeping pretrained trees frozen.
    """
    finetuned_model = XGBClassifier(
        n_estimators=n_additional_estimators,
        **pretrained_model.get_params(),
    )
    finetuned_model.fit(
        X_finetune, y_finetune,
        xgb_model=pretrained_model.get_booster(),
    )
    return finetuned_model
```

**RandomForest:**
```python
def finetune_random_forest(
    pretrained_model: RandomForestClassifier,
    X_finetune: np.ndarray,
    y_finetune: np.ndarray,
    n_additional_estimators: int = 100,
) -> RandomForestClassifier:
    """
    Fine-tune RandomForest with warm-start.

    Strategy: Use warm_start=True to add trees to existing forest.
    """
    pretrained_model.warm_start = True
    pretrained_model.n_estimators += n_additional_estimators
    pretrained_model.fit(X_finetune, y_finetune)
    return pretrained_model
```

### 7.2 NN Fine-Tuning (Full Fine-Tune with Reduced LR)

```python
def finetune_neural_network(
    pretrained_model: nn.Module,
    X_finetune: np.ndarray,
    y_finetune: np.ndarray,
    epochs: int = 50,
    lr_factor: float = 0.1,
) -> nn.Module:
    """
    Fine-tune neural network with reduced learning rate.

    Strategy:
    - Fine-tune all layers (no freezing)
    - Reduce learning rate by 10× (lr_factor = 0.1)
    - Train for 50 epochs with early stopping
    - Use same loss function and optimizer type as pretraining
    """
    # Get original learning rate from pretrained model
    original_lr = pretrained_model.original_lr  # Stored during 003b
    new_lr = original_lr * lr_factor

    # Set up optimizer with reduced LR
    optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=new_lr)

    # Fine-tune with early stopping
    # ... (standard training loop)

    return pretrained_model
```

### 7.3 Scaler Strategy

**Fine-Tuning (Transfer):**
- Reuse Berlin scaler from 003b
- Reason: Model was trained on Berlin-scaled features, expects same distribution

**From-Scratch (Leipzig-only):**
- Fit new scaler on Leipzig finetune data
- Reason: No dependency on Berlin data, optimize for Leipzig distribution

```python
# Transfer fine-tuning
X_finetune_scaled = berlin_scaler.transform(X_finetune)

# From-scratch baseline
leipzig_scaler = StandardScaler()
X_finetune_scaled = leipzig_scaler.fit_transform(X_finetune)
```

### 7.4 Class Weighting Strategy

Both fine-tuning and from-scratch training use class weighting to handle imbalance:

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_finetune),
    y=y_finetune
)
```

---

## 8. Power-Law Modeling (Improvement 6)

### 8.1 Methodology

**Power-Law Function:**
```
Performance(N) = a × N^b
```

Where:
- `N` = number of fine-tuning samples
- `a` = scaling coefficient
- `b` = exponent (determines curve shape)

**Fitting:**
```python
from scipy.optimize import curve_fit

def power_law(N, a, b):
    return a * N**b

# Fit to observed data points
params, covariance = curve_fit(
    power_law,
    n_samples_observed,
    f1_scores_observed,
    p0=[0.5, 0.3],  # Initial guess
)
a_fit, b_fit = params

# Compute R²
f1_predicted = power_law(n_samples_observed, a_fit, b_fit)
r_squared = 1 - (np.sum((f1_observed - f1_predicted)**2) /
                 np.sum((f1_observed - np.mean(f1_observed))**2))
```

### 8.2 Extrapolation to Recovery Points

**90% Recovery:**
```python
target_performance_90 = 0.90 * from_scratch_f1
n_samples_90pct = (target_performance_90 / a_fit) ** (1 / b_fit)
fraction_90pct = n_samples_90pct / total_finetune_samples
```

**95% Recovery:**
```python
target_performance_95 = 0.95 * from_scratch_f1
n_samples_95pct = (target_performance_95 / a_fit) ** (1 / b_fit)
fraction_95pct = n_samples_95pct / total_finetune_samples
```

### 8.3 Interpretation Guidelines

**Exponent `b` interpretation:**

| Exponent Range | Interpretation | Practical Meaning |
|----------------|----------------|-------------------|
| `b < 0.5` | Diminishing returns | Early data very valuable, adding more has less impact |
| `b ≈ 0.5` | Square-root scaling | Typical for many ML tasks, balanced value |
| `b > 0.5` | Accelerating returns | Later data more valuable (unusual, check for overfitting) |

**Literature Comparison:**
- Tong et al. (2019) report exponent `b ≈ 0.35` for transfer learning
- This implies 50-70% label savings compared to from-scratch training
- If our `b` is significantly different, discuss implications

### 8.4 Validation

**Quality Checks:**
1. R² > 0.90 (good fit to observed data)
2. Exponent in reasonable range (0.2 < b < 0.7)
3. Extrapolated values < 100% fraction (sanity check)
4. Visual inspection: fitted curve follows trend

**Uncertainty Quantification:**
- Report confidence intervals on `a` and `b` from covariance matrix
- Sensitivity analysis: how much does extrapolation change with ±1 std on parameters?

---

## 9. Dependencies and Outputs

### 9.1 Prerequisites

**From PRD 003b (Berlin Optimization):**
- `outputs/phase_3/models/berlin_ml_champion.pkl`
- `outputs/phase_3/models/berlin_nn_champion.pt`
- `outputs/phase_3/models/scaler.pkl` (Berlin scaler, reused for fine-tuning)

**From Phase 2 (Feature Engineering):**
- `data/phase_3_experiments/leipzig_finetune.parquet`
- `data/phase_3_experiments/leipzig_test.parquet`

**From PRD 003c (Transfer Evaluation):**
- `outputs/phase_3/metadata/transfer_evaluation.json` (zero-shot baseline)

### 9.2 Outputs

**Metadata:**
- `outputs/phase_3/metadata/finetuning_curve.json`
  - Results for ML and NN at 4 fractions
  - Power-law fit parameters (a, b, R²)
  - Recovery points (90%, 95%)
  - McNemar test results

**Figures (6 visualizations):**
- `outputs/phase_3/figures/finetuning/finetuning_curve.png`
- `outputs/phase_3/figures/finetuning/finetuning_vs_baselines.png`
- `outputs/phase_3/figures/finetuning/finetuning_per_genus_recovery.png`
- `outputs/phase_3/figures/finetuning/finetuning_ml_vs_nn_comparison.png`
- `outputs/phase_3/figures/finetuning/finetuning_significance_matrix.png`
- `outputs/phase_3/figures/finetuning/finetuning_powerlaw_extrapolation.png`

**Models:**
- `outputs/phase_3/models/finetuned/ml_champion_f010.pkl` (10% data)
- `outputs/phase_3/models/finetuned/ml_champion_f025.pkl` (25% data)
- `outputs/phase_3/models/finetuned/ml_champion_f050.pkl` (50% data)
- `outputs/phase_3/models/finetuned/ml_champion_f100.pkl` (100% data)
- `outputs/phase_3/models/finetuned/nn_champion_f010.pt` (10% data)
- `outputs/phase_3/models/finetuned/nn_champion_f025.pt` (25% data)
- `outputs/phase_3/models/finetuned/nn_champion_f050.pt` (50% data)
- `outputs/phase_3/models/finetuned/nn_champion_f100.pt` (100% data)
- `outputs/phase_3/models/leipzig_ml_scratch.pkl` (from-scratch baseline)
- `outputs/phase_3/models/leipzig_nn_scratch.pt` (from-scratch baseline)

**Logs:**
- `outputs/phase_3/logs/03d_finetuning.json`

### 9.3 Execution Context

**Runs in Parallel With:** PRD 003c (both use trained champions from 003b)

**Execution Order:**
1. PRD 003a (Setup Fixation) - produces processed datasets
2. PRD 003b (Berlin Optimization) - produces champions and scaler
3. **PRD 003c (Transfer Evaluation)** - produces zero-shot baselines
4. **PRD 003d (Fine-Tuning)** - THIS PRD

**Timeline:** Can start immediately after 003b completes (003c provides context but not hard dependency)

---

## 10. Success Criteria

### 10.1 Completeness

- [ ] Both champions (ML and NN) fine-tuned at all 4 fractions (10%, 25%, 50%, 100%)
- [ ] From-scratch baselines complete for both architectures
- [ ] All 8 fine-tuned models + 2 from-scratch models saved
- [ ] finetuning_curve.json generated with complete schema
- [ ] All 6 visualizations generated

### 10.2 Quality

- [ ] Power-law fit achieves R² > 0.90 for both champions
- [ ] Exponent `b` in reasonable range (0.2 < b < 0.7)
- [ ] 95% recovery point extrapolated successfully
- [ ] McNemar tests complete for all comparison pairs
- [ ] Per-genus recovery analysis complete for all genera

### 10.3 Scientific Validity

- [ ] Stratified sampling maintains class proportions at all fractions
- [ ] Same random seed used for both models (fair comparison)
- [ ] Berlin scaler reused for fine-tuning (correct preprocessing)
- [ ] New scaler fitted for from-scratch baselines (no data leakage)
- [ ] Statistical significance properly assessed (McNemar, p < 0.05)

### 10.4 Deliverables

- [ ] Sample efficiency curves clearly show recovery trajectory
- [ ] Power-law extrapolation visualized with uncertainty bounds
- [ ] ML vs. NN comparison identifies architectural differences
- [ ] Per-genus analysis identifies fastest-recovering genera
- [ ] From-scratch comparison quantifies transfer learning advantage
- [ ] Literature comparison contextualizes findings (Tong et al. 2019)

---

## 11. Source Module Functions

From `src/urban_tree_transfer/experiments/training.py`:

```python
def finetune_xgboost(
    pretrained_model: XGBClassifier,
    X_finetune: np.ndarray,
    y_finetune: np.ndarray,
    n_additional_estimators: int = 100,
) -> XGBClassifier:
    """Fine-tune XGBoost model with additional trees."""

def finetune_random_forest(
    pretrained_model: RandomForestClassifier,
    X_finetune: np.ndarray,
    y_finetune: np.ndarray,
    n_additional_estimators: int = 100,
) -> RandomForestClassifier:
    """Fine-tune RandomForest with warm-start."""

def finetune_neural_network(
    pretrained_model: nn.Module,
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

def fit_power_law_curve(
    n_samples: np.ndarray,
    f1_scores: np.ndarray,
) -> dict[str, float]:
    """
    Fit power-law model to sample efficiency curve.

    Returns:
        dict with keys: a, b, r_squared, formula
    """

def extrapolate_recovery_points(
    power_law_params: dict[str, float],
    from_scratch_f1: float,
    total_finetune_samples: int,
) -> dict[str, int | float]:
    """
    Extrapolate samples needed for 90% and 95% recovery.

    Returns:
        dict with keys: n_samples_90pct, n_samples_95pct,
                        fraction_to_90pct, fraction_to_95pct
    """
```

---

_Last Updated: 2026-02-07_
