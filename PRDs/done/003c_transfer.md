# PRD 003c: Transfer-Evaluation

**PRD ID:** 003c
**Status:** Draft
**Created:** 2026-02-07
**Dependencies:** PRD 003b (trained Berlin champions)

---

## 1. Overview

### 1.1 Problem Statement

After establishing optimal Berlin performance in PRD 003b, we must evaluate how well the trained models transfer to Leipzig without any target-domain fine-tuning. This zero-shot transfer evaluation quantifies the transfer gap and identifies which genera, features, and model characteristics affect cross-city generalization.

**Key Challenge:** Measuring transfer performance requires more than simple accuracy comparison. We need:
- Statistical rigor (bootstrap confidence intervals)
- Feature stability analysis (do the same features matter in both cities?)
- Per-genus analysis (which genera transfer well/poorly?)
- Hypothesis testing (validate assumptions about what drives transferability)

### 1.2 Research Question

**RQ2:** How much does performance drop when applying a Berlin-trained model to Leipzig (zero-shot)?

**Sub-questions:**
- What is the absolute and relative transfer gap?
- Which genera transfer robustly vs. poorly?
- Are the same features important in both cities?
- What factors predict transfer robustness?

### 1.3 Goals

1. **Quantify Transfer Gap:** Measure zero-shot performance degradation with bootstrap CIs
2. **Feature Stability Analysis:** Test if important features generalize across cities
3. **Per-Genus Transfer Analysis:** Identify robust and fragile genera
4. **Hypothesis Testing:** Validate a-priori assumptions about transferability drivers
5. **Select Best Transfer Model:** Choose ML or NN champion for fine-tuning (PRD 003d)

### 1.4 Non-Goals

- Fine-tuning on Leipzig data (that's PRD 003d)
- Domain adaptation methods (future work)
- Species-level transfer analysis
- Multi-city joint training

---

## 2. Experiment: 03c Transfer Evaluation

### 2.1 Purpose

Evaluate zero-shot transfer performance of both Berlin champions (ML and NN) on Leipzig test set, measure transfer gap, analyze feature stability, and test hypotheses about transferability drivers.

### 2.2 Inputs

- `data/phase_3_experiments/leipzig_test.parquet` (hold-out test set)
- `data/phase_3_experiments/leipzig_finetune.parquet` (for Leipzig from-scratch training)
- `outputs/phase_3/models/berlin_ml_champion.pkl` (trained ML model)
- `outputs/phase_3/models/berlin_nn_champion.pt` (trained NN model)
- `outputs/phase_3/models/scaler.pkl` (Berlin feature scaler)
- `outputs/phase_3/models/label_encoder.pkl` (genus label encoder)
- `outputs/phase_3/metadata/berlin_evaluation.json` (Berlin test metrics)
- `outputs/phase_3/metadata/setup_decisions.json` (CHM, features config)

### 2.3 Processing Steps

#### Step 1: Load Models and Data

- Load both trained Berlin champions (ML and NN)
- Load Leipzig test set (hold-out data, never seen during training)
- Load Berlin scaler and label encoder
- Apply identical preprocessing (scaler fit on Berlin, applied to Leipzig)

**Critical:** Use Berlin scaler on Leipzig data (zero-shot = no target-domain data access)

#### Step 2: Zero-Shot Evaluation

- Predict on Leipzig test with both models
- Compute comprehensive metrics:
  - Overall: Weighted F1, Macro F1, Accuracy
  - Per-genus: F1, Precision, Recall, Support
- **NEW (Imp 4):** All metrics with **bootstrap confidence intervals** (1000 resamples, 95% CI)

#### Step 3: Transfer Gap Analysis

Compute transfer-specific metrics for each model:

1. **Absolute Drop:** `F1_Berlin - F1_Leipzig`
2. **Relative Drop:** `(F1_Berlin - F1_Leipzig) / F1_Berlin * 100`
3. **Statistical Significance Test (Imp 4):**
   - Mann-Whitney U test on bootstrap distributions
   - H0: No transfer gap (Berlin = Leipzig performance)
   - Report p-value and effect size
4. **Model Comparison:** ML vs. NN transfer robustness

#### Step 4: Leipzig From-Scratch Training (Imp 3)

**Purpose:** Feature importance comparison for stability analysis

- Train ML champion on Leipzig finetune set
- Use identical hyperparameters as Berlin champion (from 03b HP tuning)
- Fit new scaler on Leipzig data (independent from Berlin)
- Evaluate on Leipzig test set
- Extract feature importances

**Rationale:** Compare feature rankings between Berlin and Leipzig models to test if the same features are important in both cities.

#### Step 5: Feature Stability Analysis (Imp 3)

Quantify feature importance generalization:

1. **Extract Importances:**
   - Berlin champion: feature importances on Berlin train data
   - Leipzig from-scratch: feature importances on Leipzig train data
2. **Compute Spearman Rank Correlation (ρ):**
   - ρ > 0.7 = high stability (same features matter)
   - ρ < 0.5 = low stability (city-specific features dominate)
3. **Identify Stable/Unstable Features:**
   - Most stable: near diagonal in scatter plot
   - Most unstable: far from diagonal
4. **Literature Validation:**
   - Red-Edge indices expected stable (Immitzer 2019)
   - CHM expected city-specific (urban structure differs)

#### Step 6: Per-Genus Transfer Analysis with A-Priori Hypotheses (Imp 5)

**CRITICAL:** The following 4 hypotheses must be documented in `docs/documentation/03_Experiments/03_Transfer_Evaluation.md` BEFORE running 03c to avoid post-hoc bias.

##### H1: Sample Size Hypothesis

- **Claim:** Genera with more Berlin training samples transfer better
- **Test:** Pearson correlation between `berlin_sample_count` and `transfer_gap`
- **Expected:** r < 0 (negative correlation = more samples → smaller gap)
- **Rationale:** Larger training sets provide better feature coverage

##### H2: Conifer vs. Deciduous Hypothesis

- **Claim:** Nadelbäume have lower transfer gap than Laubbäume
- **Test:** Mann-Whitney U between conifer and deciduous transfer gaps
- **Rationale:** Nadelbäume have more distinct spectral profile (Fassnacht 2016)
- **Groups:**
  - Nadelbäume: PINUS, PICEA
  - Laubbäume: all others

##### H3: Phenological Distinctness Hypothesis

- **Claim:** Genera with early leaf-out (BETULA, SALIX) have higher transfer gap
- **Test:** Compare transfer gaps for early vs. mid-season genera
- **Rationale:** Regional phenological differences (Hemmerling 2021)
- **Groups:**
  - Early: BETULA, SALIX
  - Mid-season: others

##### H4: Red-Edge Robustness Hypothesis

- **Claim:** Genera with high Red-Edge feature importance transfer better
- **Test:** Correlation between Red-Edge importance and transfer robustness
- **Rationale:** Red-Edge indices optimal for tree species (Immitzer 2019)
- **Metric:** Spearman ρ between genus-level Red-Edge importance and transfer F1

**Output:** Statistical test results for all 4 hypotheses with p-values, effect sizes, and interpretation.

#### Step 7: Transfer Robustness Classification

Classify each genus by transfer performance:

1. **Compute Per-Genus F1 Comparison:**
   - Berlin F1 (from berlin_evaluation.json)
   - Leipzig F1 (from zero-shot evaluation)
   - Absolute drop
   - Relative drop percentage
2. **Robustness Categories:**
   - **Robust:** <5% relative drop
   - **Medium:** 5-15% relative drop
   - **Poor:** >15% relative drop
3. **Annotate with Context:**
   - Berlin sample count
   - Nadel/Laub group
   - Hypothesis test results

#### Step 8: Confusion Matrix Comparison

- Generate side-by-side confusion matrices:
  - Berlin test confusion (from 03b)
  - Leipzig test confusion (from 03c)
- Highlight systematic differences
- Identify most confused genus pairs in Leipzig vs. Berlin
- Use German genus labels (genus_german)

#### Step 9: Extended Transfer Analysis

1. **Nadel vs. Laub Aggregate:**
   - Average transfer gap per group
   - Test for significant difference
2. **Species-Level Detail:**
   - For genera with poor transfer (F1 drop >15%)
   - Show species-level distribution in both cities
   - Identify if species composition differs
3. **Cross-City Comparison:**
   - Are the same genera problematic in both cities?
   - Are confusion patterns similar?

#### Step 10: Select Best Transfer Model

- Compare ML and NN zero-shot transfer performance
- Selection criteria:
  - Leipzig test F1 (higher = better)
  - Transfer gap (smaller = better)
  - Per-genus robustness (more robust genera = better)
- Select model for fine-tuning experiments (PRD 003d)

### 2.4 Outputs

**Metadata:**
- `outputs/phase_3/metadata/transfer_evaluation.json`
  - Zero-shot metrics for ML and NN
  - Transfer gap analysis (absolute, relative, significance)
  - Per-genus transfer robustness
  - Feature stability analysis (Spearman ρ, stable/unstable features)
  - Hypothesis test results (H1-H4)
  - Leipzig from-scratch metrics
  - Best transfer model selection + reasoning

**Figures:** `outputs/phase_3/figures/transfer/`
- `transfer_comparison.png` (with CI error bars - Imp 4)
- `confusion_comparison_berlin_leipzig.png` (side-by-side)
- `per_genus_transfer_robustness.png` (bar chart with robustness categories)
- `transfer_conifer_deciduous.png` (Nadel/Laub aggregate)
- `transfer_confusion_pairs.png` (most confused pairs in Leipzig)
- `transfer_species_analysis.png` (species detail for poor genera)
- **NEW (Imp 3):** `feature_stability_scatter.png` (Berlin vs. Leipzig feature importance scatter with feature-type colors)
- **NEW (Imp 5):** `transfer_robustness_ranking.png` (horizontal bar with sample size annotations)
- **NEW (Imp 5):** `hypothesis_test_summary.png` (4 hypothesis results with p-values)

**Logs:**
- `outputs/phase_3/logs/03c_transfer_evaluation.json`

---

## 3. Data Flow

```
Trained Champions (from 003b)           Setup Decisions (from 003a)
├── berlin_ml_champion.pkl              ├── setup_decisions.json
├── berlin_nn_champion.pt               └── Leipzig splits (from Phase 2)
├── scaler.pkl                              ├── leipzig_finetune.parquet
├── label_encoder.pkl                       └── leipzig_test.parquet
└── berlin_evaluation.json
         │                                           │
         └──────────────┬────────────────────────────┘
                        │
                        ▼
              ┌─────────────────────────┐
              │ 03c_transfer_evaluation │
              └─────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
   Metadata         Figures         Logs
transfer_evaluation  transfer/  03c_transfer_
    .json           *.png       evaluation.json
```

**Parallel Experiments:**
- PRD 003c (transfer evaluation) and PRD 003d (fine-tuning) both use the trained champions from PRD 003b
- PRD 003c must complete first to inform PRD 003d (select best transfer model)

---

## 4. Configuration

### 4.1 Transfer Evaluation Config

```yaml
# configs/experiments/phase3_config.yaml (excerpt)

transfer_evaluation:
  # Bootstrap confidence intervals
  confidence_intervals: true
  n_bootstrap: 1000
  ci_level: 0.95

  # Transfer gap thresholds
  robustness_thresholds:
    robust: 0.05    # <5% relative drop
    medium: 0.15    # 5-15% relative drop
    # poor: >15% relative drop

  # Feature stability
  feature_stability:
    method: spearman  # rank correlation
    stability_threshold: 0.7  # ρ > 0.7 = high stability

  # Statistical tests
  statistical_tests:
    transfer_gap_test: mann_whitney_u  # H0: no gap
    hypothesis_tests:
      - h1_sample_size  # Pearson correlation
      - h2_conifer_deciduous  # Mann-Whitney U
      - h3_phenological  # Mann-Whitney U
      - h4_red_edge  # Spearman correlation

  # A-priori hypothesis groups
  hypothesis_groups:
    conifer_genera: [PINUS, PICEA]
    early_leafout_genera: [BETULA, SALIX]
    red_edge_features:
      - NDVI_RE1
      - NDVI_RE2
      - NDVI_RE3
      - CIre
      - NDVIre1n
      - NDVIre2n
      - NDVIre3n
```

### 4.2 Leipzig From-Scratch Training Config

```yaml
leipzig_from_scratch:
  # Use identical HP as Berlin champion
  use_berlin_hp: true

  # Train on finetune split (same size as Berlin train)
  dataset: leipzig_finetune

  # Fit new scaler (independent from Berlin)
  fit_scaler: true

  # Purpose: feature importance comparison only
  save_model: false  # not needed for downstream tasks
```

---

## 5. JSON Schema

### 5.1 Transfer Evaluation Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Transfer Evaluation",
  "type": "object",
  "required": ["timestamp", "source_city", "target_city", "models"],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "source_city": { "type": "string", "enum": ["berlin"] },
    "target_city": { "type": "string", "enum": ["leipzig"] },
    "models": {
      "type": "object",
      "description": "Transfer metrics for each model (ml_champion, nn_champion)",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "source_f1": {
            "type": "number",
            "description": "Berlin test F1 (from berlin_evaluation.json)"
          },
          "source_f1_ci_lower": { "type": "number" },
          "source_f1_ci_upper": { "type": "number" },
          "target_f1": {
            "type": "number",
            "description": "Leipzig test F1 (zero-shot)"
          },
          "target_f1_ci_lower": { "type": "number" },
          "target_f1_ci_upper": { "type": "number" },
          "absolute_drop": {
            "type": "number",
            "description": "F1_Berlin - F1_Leipzig"
          },
          "relative_drop_pct": {
            "type": "number",
            "description": "(F1_Berlin - F1_Leipzig) / F1_Berlin * 100"
          },
          "transfer_gap_significance": {
            "type": "object",
            "description": "Statistical test for transfer gap (Imp 4)",
            "properties": {
              "test": { "type": "string", "const": "mann_whitney_u" },
              "p_value": { "type": "number" },
              "significant": { "type": "boolean" },
              "effect_size": { "type": "number" }
            }
          },
          "per_genus_transfer": {
            "type": "object",
            "description": "Per-genus transfer analysis",
            "additionalProperties": {
              "type": "object",
              "properties": {
                "source_f1": { "type": "number" },
                "target_f1": { "type": "number" },
                "drop": { "type": "number" },
                "drop_pct": { "type": "number" },
                "robustness": {
                  "type": "string",
                  "enum": ["robust", "medium", "poor"]
                },
                "berlin_sample_count": { "type": "integer" },
                "leipzig_sample_count": { "type": "integer" }
              }
            }
          },
          "nadel_laub_analysis": {
            "type": "object",
            "description": "Aggregate transfer by conifer/deciduous",
            "properties": {
              "nadel_gap_pct": { "type": "number" },
              "laub_gap_pct": { "type": "number" },
              "difference": { "type": "number" },
              "p_value": { "type": "number" }
            }
          }
        }
      }
    },
    "best_transfer_model": {
      "type": "string",
      "enum": ["ml_champion", "nn_champion"]
    },
    "selection_reasoning": {
      "type": "string",
      "description": "Why this model was selected for fine-tuning"
    },
    "leipzig_from_scratch": {
      "type": "object",
      "description": "Leipzig-only training for feature stability (Imp 3)",
      "properties": {
        "leipzig_f1": { "type": "number" },
        "leipzig_f1_ci_lower": { "type": "number" },
        "leipzig_f1_ci_upper": { "type": "number" },
        "feature_importances": {
          "type": "object",
          "additionalProperties": { "type": "number" }
        }
      }
    },
    "feature_stability": {
      "type": "object",
      "description": "Feature importance stability analysis (Imp 3)",
      "properties": {
        "spearman_rho": {
          "type": "number",
          "description": "Rank correlation between Berlin and Leipzig feature importances"
        },
        "interpretation": {
          "type": "string",
          "enum": ["high_stability", "moderate_stability", "low_stability"]
        },
        "most_stable_features": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Top-5 features with most consistent ranking"
        },
        "most_unstable_features": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Top-5 features with most different ranking"
        },
        "literature_validation": {
          "type": "string",
          "description": "Do stable features match literature expectations?"
        }
      }
    },
    "hypothesis_tests": {
      "type": "array",
      "description": "A-priori hypothesis test results (Imp 5)",
      "items": {
        "type": "object",
        "properties": {
          "hypothesis_id": {
            "type": "string",
            "enum": ["H1", "H2", "H3", "H4"]
          },
          "description": { "type": "string" },
          "test_statistic": {
            "type": "string",
            "description": "E.g., 'pearson_r', 'mann_whitney_u', 'spearman_rho'"
          },
          "test_value": {
            "type": "number",
            "description": "Computed statistic (r, U, ρ, etc.)"
          },
          "p_value": { "type": "number" },
          "result": {
            "type": "string",
            "enum": ["confirmed", "rejected", "inconclusive"],
            "description": "confirmed if p < 0.05 and expected direction"
          },
          "effect_size": {
            "type": "number",
            "description": "Cohen's d or similar"
          }
        }
      }
    }
  }
}
```

---

## 6. Visualizations

### 6.1 Required Figures

All figures use German genus labels (genus_german) and publication-ready formatting.

| Figure | Filename | Description | Improvement |
|--------|----------|-------------|-------------|
| Transfer F1 comparison | `transfer_comparison.png` | Bar chart comparing Berlin vs. Leipzig F1 for both models with bootstrap CI error bars | Imp 4 |
| Confusion comparison | `confusion_comparison_berlin_leipzig.png` | Side-by-side confusion matrices (Berlin test vs. Leipzig test) with German labels | - |
| Per-genus transfer robustness | `per_genus_transfer_robustness.png` | Bar chart showing F1 drop per genus with robustness color coding | - |
| Nadel/Laub transfer | `transfer_conifer_deciduous.png` | Aggregate transfer gap comparison for conifers vs. deciduous | - |
| Most confused pairs | `transfer_confusion_pairs.png` | Top-10 most confused genus pairs in Leipzig (bar chart) | - |
| Species analysis | `transfer_species_analysis.png` | Species-level breakdown for genera with poor transfer (>15% drop) | - |
| Feature stability scatter | `feature_stability_scatter.png` | Scatter plot: Berlin feature importance (x) vs. Leipzig feature importance (y) with feature-type colors and diagonal reference line | Imp 3 |
| Transfer robustness ranking | `transfer_robustness_ranking.png` | Horizontal bar chart ranked by transfer gap with sample size annotations | Imp 5 |
| Hypothesis test summary | `hypothesis_test_summary.png` | 4-panel figure showing results for H1-H4 with p-values and effect sizes | Imp 5 |

### 6.2 Visualization Functions

```python
# src/urban_tree_transfer/experiments/visualization.py

def plot_transfer_comparison(
    berlin_metrics: dict[str, float],
    leipzig_metrics: dict[str, float],
    berlin_ci: dict[str, tuple[float, float]],
    leipzig_ci: dict[str, tuple[float, float]],
    output_path: Path,
) -> None:
    """Bar chart comparing Berlin vs. Leipzig F1 with CI error bars."""

def plot_confusion_comparison(
    berlin_cm: np.ndarray,
    leipzig_cm: np.ndarray,
    labels: list[str],  # German names
    output_path: Path,
) -> None:
    """Side-by-side confusion matrices."""

def plot_per_genus_transfer(
    transfer_results: dict[str, dict[str, Any]],
    genus_labels: list[str],  # German names
    output_path: Path,
) -> None:
    """Per-genus transfer robustness with color coding."""

def plot_feature_stability(
    berlin_importances: dict[str, float],
    leipzig_importances: dict[str, float],
    feature_types: dict[str, str],  # spectral, chm, proximity, etc.
    spearman_rho: float,
    output_path: Path,
) -> None:
    """Scatter plot of feature importance stability (Imp 3)."""

def plot_hypothesis_tests(
    hypothesis_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """4-panel summary of hypothesis test results (Imp 5)."""
```

---

## 7. Source Module Functions

### 7.1 Transfer Metrics (transfer.py)

```python
def compute_transfer_metrics(
    source_metrics: dict[str, Any],
    target_metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute transfer-specific metrics.

    Args:
        source_metrics: Berlin evaluation results
        target_metrics: Leipzig evaluation results

    Returns:
        dict with:
        - absolute_drop: F1_Berlin - F1_Leipzig
        - relative_drop_pct: (drop / F1_Berlin) * 100
        - transfer_gap_significance: Mann-Whitney U test result
    """

def compute_per_genus_transfer(
    source_per_genus: dict[str, float],
    target_per_genus: dict[str, float],
    berlin_sample_counts: dict[str, int],
    leipzig_sample_counts: dict[str, int],
    thresholds: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """
    Compute per-genus transfer analysis with robustness classification.

    Args:
        source_per_genus: Berlin F1 per genus
        target_per_genus: Leipzig F1 per genus
        berlin_sample_counts: Training samples per genus in Berlin
        leipzig_sample_counts: Test samples per genus in Leipzig
        thresholds: Robustness category thresholds

    Returns:
        dict mapping genus -> {
            source_f1, target_f1, drop, drop_pct,
            robustness, berlin_sample_count, leipzig_sample_count
        }
    """

def compute_feature_stability(
    berlin_importances: dict[str, float],
    leipzig_importances: dict[str, float],
) -> dict[str, Any]:
    """
    Compute feature importance stability (Imp 3).

    Args:
        berlin_importances: Feature importances from Berlin model
        leipzig_importances: Feature importances from Leipzig model

    Returns:
        dict with:
        - spearman_rho: Rank correlation
        - interpretation: 'high', 'moderate', or 'low' stability
        - most_stable_features: Top-5 consistent features
        - most_unstable_features: Top-5 different features
        - literature_validation: Red-Edge vs. CHM comparison
    """

def test_transfer_hypotheses(
    per_genus_transfer: dict[str, dict[str, Any]],
    berlin_feature_importances: dict[str, float],
    genus_metadata: pd.DataFrame,  # with is_conifer, phenology columns
) -> list[dict[str, Any]]:
    """
    Test a-priori transfer hypotheses (Imp 5).

    Args:
        per_genus_transfer: Per-genus transfer results
        berlin_feature_importances: Feature importances
        genus_metadata: Genus-level metadata

    Returns:
        List of 4 hypothesis test results (H1-H4)
    """

def create_transfer_summary(
    ml_transfer: dict[str, Any],
    nn_transfer: dict[str, Any],
) -> dict[str, Any]:
    """
    Create summary comparing ML and NN transfer performance.

    Returns:
        dict with:
        - best_transfer_model: 'ml_champion' or 'nn_champion'
        - selection_reasoning: why this model was selected
    """
```

---

## 8. Success Criteria

1. **Zero-Shot Evaluation Complete:**
   - Both Berlin champions evaluated on Leipzig test
   - All metrics computed with bootstrap CIs (1000 resamples)
   - Transfer gap statistically significant (p < 0.05)

2. **Feature Stability Quantified:**
   - Spearman ρ computed between Berlin and Leipzig feature importances
   - Stable and unstable features identified
   - Literature validation documented (Red-Edge vs. CHM)

3. **Per-Genus Transfer Analysis:**
   - All genera classified as robust/medium/poor
   - Sample sizes annotated
   - Nadel/Laub aggregate comparison complete

4. **Hypothesis Testing Complete:**
   - All 4 a-priori hypotheses tested (H1-H4)
   - P-values and effect sizes reported
   - Results interpreted (confirmed/rejected/inconclusive)

5. **Best Transfer Model Selected:**
   - ML vs. NN comparison documented
   - Selection reasoning justified
   - Model ready for PRD 003d (fine-tuning)

6. **All Outputs Generated:**
   - `transfer_evaluation.json` validates against schema
   - All 9 figures generated and publication-ready
   - Logs complete

---

## 9. Dependencies and Prerequisites

### 9.1 Prerequisites

**From PRD 003b (Berlin Optimization):**
- `outputs/phase_3/models/berlin_ml_champion.pkl`
- `outputs/phase_3/models/berlin_nn_champion.pt`
- `outputs/phase_3/models/scaler.pkl`
- `outputs/phase_3/models/label_encoder.pkl`
- `outputs/phase_3/metadata/berlin_evaluation.json`

**From PRD 003a (Setup Fixation):**
- `outputs/phase_3/metadata/setup_decisions.json` (CHM config, feature set)

**From Phase 2 (Feature Engineering):**
- `data/phase_3_experiments/leipzig_test.parquet` (hold-out test split)
- `data/phase_3_experiments/leipzig_finetune.parquet` (for Leipzig from-scratch training)

### 9.2 Outputs

**Metadata:**
- `outputs/phase_3/metadata/transfer_evaluation.json`

**Figures:**
- `outputs/phase_3/figures/transfer/transfer_comparison.png`
- `outputs/phase_3/figures/transfer/confusion_comparison_berlin_leipzig.png`
- `outputs/phase_3/figures/transfer/per_genus_transfer_robustness.png`
- `outputs/phase_3/figures/transfer/transfer_conifer_deciduous.png`
- `outputs/phase_3/figures/transfer/transfer_confusion_pairs.png`
- `outputs/phase_3/figures/transfer/transfer_species_analysis.png`
- `outputs/phase_3/figures/transfer/feature_stability_scatter.png` (Imp 3)
- `outputs/phase_3/figures/transfer/transfer_robustness_ranking.png` (Imp 5)
- `outputs/phase_3/figures/transfer/hypothesis_test_summary.png` (Imp 5)

**Logs:**
- `outputs/phase_3/logs/03c_transfer_evaluation.json`

### 9.3 Parallel Experiments

- **PRD 003c (this document):** Transfer evaluation
- **PRD 003d (next):** Fine-tuning experiments
- **Relationship:** PRD 003c selects best transfer model → PRD 003d uses it for fine-tuning

---

## 10. A-Priori Hypotheses

**CRITICAL:** These hypotheses must be documented in `docs/documentation/03_Experiments/03_Transfer_Evaluation.md` BEFORE running 03c to avoid post-hoc bias.

### H1: Sample Size Hypothesis

**Claim:** Genera with more Berlin training samples transfer better (smaller transfer gap)

**Rationale:** Larger training sets provide better feature coverage and reduce overfitting to source-domain artifacts.

**Test:**
- Metric: Pearson correlation
- Variables: `berlin_sample_count` (x) vs. `transfer_gap` (y)
- Expected: r < 0 (negative correlation)
- Significance: p < 0.05

### H2: Conifer vs. Deciduous Hypothesis

**Claim:** Nadelbäume have lower transfer gap than Laubbäume

**Rationale:** Nadelbäume have more distinct and stable spectral signatures compared to deciduous trees (Fassnacht 2016).

**Test:**
- Metric: Mann-Whitney U test
- Groups:
  - Nadelbäume: PINUS, PICEA
  - Laubbäume: all others
- Expected: median(nadel_gap) < median(laub_gap)
- Significance: p < 0.05

### H3: Phenological Distinctness Hypothesis

**Claim:** Genera with early leaf-out (BETULA, SALIX) have higher transfer gap

**Rationale:** Regional phenological differences cause spectral shifts. Early-leafing species show more variation across cities (Hemmerling 2021).

**Test:**
- Metric: Mann-Whitney U test
- Groups:
  - Early leaf-out: BETULA, SALIX
  - Mid-season leaf-out: others
- Expected: median(early_gap) > median(mid_gap)
- Significance: p < 0.05

### H4: Red-Edge Robustness Hypothesis

**Claim:** Genera with high Red-Edge feature importance transfer better

**Rationale:** Red-Edge indices are optimal for tree species classification and less sensitive to atmospheric/illumination variations (Immitzer 2019).

**Test:**
- Metric: Spearman rank correlation
- Variables: genus-level Red-Edge importance (x) vs. transfer F1 (y)
- Expected: ρ > 0 (positive correlation)
- Significance: p < 0.05

---

## 11. Methodological Notes

### 11.1 Bootstrap Confidence Intervals

- Method: Non-parametric bootstrap (1000 resamples)
- CI Level: 95% (2.5th and 97.5th percentiles)
- Applied to: Overall F1, per-genus F1, transfer gaps
- Rationale: Provides variance estimates without multi-seed training

### 11.2 Statistical Significance Testing

- **Transfer Gap Test:** Mann-Whitney U (H0: no gap)
- **Hypothesis Tests:** Pearson r, Mann-Whitney U, Spearman ρ as appropriate
- **Significance Level:** α = 0.05
- **Effect Sizes:** Cohen's d or rank-biserial correlation

### 11.3 Feature Stability

- **Method:** Spearman rank correlation (robust to non-linear scaling)
- **Interpretation:**
  - ρ > 0.7: High stability (same features matter)
  - 0.5 < ρ < 0.7: Moderate stability
  - ρ < 0.5: Low stability (city-specific features)

### 11.4 Zero-Shot Transfer Setup

- **Preprocessing:** Berlin scaler applied to Leipzig data (no target-domain fitting)
- **No Fine-Tuning:** Models frozen after Berlin training
- **Hold-Out Test:** Leipzig test set never seen during training or validation

---

## 12. References

- Fassnacht et al. (2016): Review of tree species classification (conifers vs. deciduous)
- Immitzer et al. (2019): Sentinel-2 Red-Edge for tree species (optimal features)
- Hemmerling et al. (2021): Phenological timing and regional variation

---

_Last Updated: 2026-02-07_
