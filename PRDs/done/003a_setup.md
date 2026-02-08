# PRD 003a: Setup-Fixierung

**PRD ID:** 003a
**Status:** Draft
**Created:** 2026-02-07
**Dependencies:** Phase 2 outputs only
**Next PRD:** 003b (Berlin-Optimierung)

---

## 1. Overview

### 1.1 Problem Statement

Before conducting systematic model comparisons and hyperparameter tuning, we must fix the experimental setup to ensure fair and reproducible comparisons. This includes deciding:

1. **CHM Strategy:** Which (if any) CHM features to include
2. **Proximity Filter:** Whether to use proximity-filtered datasets
3. **Outlier Removal:** Whether and how aggressively to remove outliers
4. **Feature Reduction:** How many features to use

These decisions must be made using only Berlin data to avoid information leakage from the target domain (Leipzig). All decisions are documented in `setup_decisions.json` for reproducibility and traceability.

### 1.2 Research Questions

| ID      | Question                                                              | Experiment    |
| ------- | --------------------------------------------------------------------- | ------------- |
| **SQ1** | Which CHM features (if any) improve Berlin performance without overfitting? | exp_08        |
| **SQ2** | Does proximity filtering improve classification quality?              | exp_08b       |
| **SQ3** | Does outlier removal improve model generalization?                    | exp_08c       |
| **SQ4** | What is the optimal feature count (F1 vs. complexity trade-off)?     | exp_09        |

### 1.3 Goals

1. **Systematic Ablation:** Test each setup dimension independently
2. **Data-Driven Decisions:** Use objective criteria, not assumptions
3. **Documented Rationale:** Log decision logic for reproducibility
4. **Transfer-Safe Setup:** Avoid target-domain information leakage
5. **Pareto Efficiency:** Balance performance with simplicity

### 1.4 Non-Goals

- Model algorithm comparison (PRD 003b)
- Hyperparameter tuning (PRD 003b)
- Transfer evaluation (PRD 003c)
- Fine-tuning experiments (PRD 003d)

---

## 2. Architecture

### 2.1 Directory Structure

```
urban-tree-transfer/
├── configs/
│   └── experiments/
│       └── phase3_config.yaml               # Setup ablation configs (Section 3)
├── src/urban_tree_transfer/
│   └── experiments/
│       ├── ablation.py                      # Ablation utilities
│       └── visualization.py                 # Standardized plots
├── notebooks/
│   └── exploratory/
│       ├── exp_08_chm_ablation.ipynb        # CHM strategy decision
│       ├── exp_08b_proximity_ablation.ipynb # Proximity filter decision
│       ├── exp_08c_outlier_ablation.ipynb   # Outlier removal decision
│       └── exp_09_feature_reduction.ipynb   # Feature selection
├── schemas/
│   └── setup_decisions.schema.json          # JSON schema validation
└── outputs/
    └── phase_3/
        ├── metadata/
        │   └── setup_decisions.json         # Final setup decisions
        └── figures/
            ├── exp_08_chm_ablation/         # CHM decision plots
            ├── exp_08b_proximity/           # Proximity filter plots
            ├── exp_08c_outlier/             # Outlier removal plots
            └── exp_09_feature_reduction/    # Feature selection plots
```

### 2.2 Data Flow

```
Phase 2 Outputs (data/phase_2_splits/)
├── berlin_train.parquet (70%)
├── berlin_val.parquet (15%)
├── berlin_train_filtered.parquet (proximity-filtered variant)
└── berlin_val_filtered.parquet
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  exp_08_chm_ablation.ipynb                                  │
│  ├── Compare: No CHM vs. zscore vs. percentile vs. both     │
│  ├── Feature importance analysis per variant                │
│  └── Decision logging                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            setup_decisions.json (chm_strategy)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  exp_08b_proximity_ablation.ipynb                           │
│  ├── Compare: Baseline vs. Proximity-filtered datasets      │
│  ├── 3-Fold Spatial Block CV with RF                        │
│  ├── Dataset size impact analysis                           │
│  └── Decision logging                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            setup_decisions.json (+ proximity_strategy)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  exp_08c_outlier_ablation.ipynb                             │
│  ├── Compare: No removal vs. high vs. high+medium           │
│  ├── 3-Fold Spatial Block CV with RF                        │
│  ├── Sample size vs. F1 trade-off                           │
│  └── Decision logging                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            setup_decisions.json (+ outlier_strategy)
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
            setup_decisions.json (COMPLETE)
            - chm_strategy
            - proximity_strategy
            - outlier_strategy
            - feature_set
            - selected_features
```

---

## 3. Configuration

### 3.1 Setup Ablation Config (from `configs/experiments/phase3_config.yaml`)

```yaml
# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
global:
  random_seed: 42
  n_jobs: -1 # Use all cores
  cv_folds: 3 # 3-fold for all CV during training/tuning (speed)
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
  ci_method: bootstrap
  n_bootstrap: 1000

# =============================================================================
# PHASE 3.1: SETUP ABLATION
# =============================================================================
setup_ablation:
  # CHM Strategy Comparison
  chm:
    features:
      - name: CHM_1m
        description: "Raw CHM height (absolute meters, known overfitting risk)"
      - name: CHM_1m_zscore
        description: "Z-score normalized within genus×city"
      - name: CHM_1m_percentile
        description: "Percentile rank within genus×city (0-100)"

    # Ablation variants for comparison
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

    # Per-feature decision criteria
    decision_rules:
      importance_threshold: 0.25 # Single feature >25% importance = problematic
      min_improvement: 0.03 # Must improve F1 >3% to justify inclusion
      max_gap_increase: 0.05 # Must not increase Train-Val gap by >5pp
      prefer_simpler: true # Prefer fewer CHM features if difference marginal

  # Proximity Filter Ablation
  proximity_filter:
    description: "Compare baseline vs. proximity-filtered datasets"
    variants:
      - name: baseline
        dataset_suffix: "" # berlin_train.parquet etc.
        description: "All trees, no proximity filter"
      - name: filtered
        dataset_suffix: "_filtered" # berlin_train_filtered.parquet etc.
        description: "Trees with ≥20m distance to different genus"

    # Decision criteria
    decision_rules:
      min_improvement: 0.02 # Filtered must improve F1 by >2pp
      max_sample_loss: 0.20 # Filtered must not lose >20% of samples
      prefer_larger_dataset: true # If F1 difference marginal, keep baseline

  # Outlier Removal Ablation
  outlier_removal:
    description: "Determine outlier removal strategy using severity flags"
    filter_column: outlier_severity # From Phase 2 quality flags
    count_column: outlier_method_count # Number of methods flagging as outlier (0-3)
    variants:
      - name: no_removal
        remove_levels: [] # Keep all samples
        description: "No outlier removal"
      - name: remove_high
        remove_levels: [high] # Remove only high-severity outliers
        description: "Remove trees flagged as high severity (≥2 methods)"
      - name: remove_high_medium
        remove_levels: [high, medium] # Remove high + medium
        description: "Remove trees flagged as high or medium severity"

    # Decision criteria
    decision_rules:
      min_improvement: 0.02 # Removal must improve F1 by >2pp
      max_sample_loss: 0.15 # Must not lose >15% of training samples
      prefer_no_removal: true # If tied, keep all data

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
```

---

## 4. Experiments

### 4.1 Exploratory Notebook: exp_08_chm_ablation.ipynb

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
   - 3-Fold Spatial Block CV
   - Record: F1, Train-Val Gap, Feature Importance

3. **Per-Feature Importance & Gap Analysis**
   - For each variant with CHM: compute per-feature importance
   - For each variant: measure Train-Val Gap increase vs. no_chm baseline
   - Evaluate each CHM feature (raw, zscore, percentile) independently

4. **Per-Feature Decision**
   - Apply decision rules to each CHM feature independently
   - Determine which (if any) CHM features to include
   - Document reasoning per feature

**Decision Logic (per CHM feature):**

```python
# Pseudocode — each CHM feature evaluated independently
baseline_f1 = no_chm_val_f1
baseline_gap = no_chm_train_val_gap

included_features = []
for feature in ["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"]:
    variant = get_variant_with(feature)

    # Criterion 1: Feature must not dominate model
    if feature_importance(feature) > 0.25:
        exclude(feature, reason="dominates model, overfitting risk")
        continue

    # Criterion 2: Must not destabilize generalization
    if variant.train_val_gap - baseline_gap > 0.05:
        exclude(feature, reason="increases Train-Val gap by >5pp")
        continue

    # Criterion 3: Must provide meaningful improvement
    if variant.val_f1 - baseline_f1 < 0.03:
        exclude(feature, reason="marginal improvement <3pp")
        continue

    included_features.append(feature)

decision = included_features if included_features else "no_chm"
```

**Note:** No Leipzig data is used in this decision. Transfer effects of CHM are evaluated later in PRD 003c (transfer evaluation). Using target-domain data in setup decisions would constitute an information leak.

**Outputs:**

- `outputs/phase_3/metadata/setup_decisions.json` (partial: chm_strategy)
- `outputs/phase_3/figures/exp_08_chm_ablation/*.png`

**Visualizations:**

| Experiment | Plot Description                | Filename                            |
| ---------- | ------------------------------- | ----------------------------------- |
| exp_08     | CHM ablation F1 comparison      | `chm_ablation_results.png`          |
| exp_08     | CHM feature importance          | `chm_feature_importance.png`        |
| exp_08     | CHM Train-Val gap comparison    | `chm_train_val_gap.png`             |

---

### 4.2 Exploratory Notebook: exp_08b_proximity_ablation.ipynb

**Purpose:** Determine whether proximity-filtered datasets improve classification

Phase 2c created two dataset variants:

- **Baseline:** All trees meeting genus frequency threshold
- **Filtered:** Trees with ≥20m distance to nearest tree of a different genus (reduces label noise from mixed crowns)

This ablation determines which variant to use for all subsequent experiments.

**Inputs:**

- `data/phase_2_splits/berlin_train.parquet` (baseline)
- `data/phase_2_splits/berlin_train_filtered.parquet` (filtered)
- `data/phase_2_splits/berlin_val.parquet` / `berlin_val_filtered.parquet`
- `outputs/phase_3/metadata/setup_decisions.json` (chm_strategy from exp_08)

**Key Tasks:**

1. **Load Both Variants**
   - Apply CHM decision from exp_08 (consistent feature set)
   - Report sample counts: baseline N vs. filtered N

2. **Cross-Validation Comparison**
   - Use Random Forest with default HP (stable baseline)
   - 3-Fold Spatial Block CV on each variant
   - Record: F1, Per-Genus F1, Train-Val Gap

3. **Sample Loss Analysis**
   - Compute: `sample_loss_pct = 1 - (N_filtered / N_baseline)`
   - Per-genus breakdown: which genera lose the most samples?
   - Identify if any genus drops below minimum viable count

4. **Decision**
   - Apply decision rules from config
   - Document trade-off (F1 gain vs. sample loss)

**Decision Logic:**

```python
# Pseudocode
baseline_f1 = cv_f1(baseline_dataset)
filtered_f1 = cv_f1(filtered_dataset)
sample_loss = 1 - (len(filtered) / len(baseline))

if sample_loss > 0.20:
    decision = "baseline"
    reason = f"Filtered loses {sample_loss:.0%} of samples — too much"
elif filtered_f1 - baseline_f1 < 0.02:
    decision = "baseline"
    reason = f"Improvement only {filtered_f1 - baseline_f1:.3f} — marginal"
else:
    decision = "filtered"
    reason = f"Filtered improves F1 by {filtered_f1 - baseline_f1:.3f} with acceptable sample loss"
```

**Note:** Same information-leak principle as CHM — only Berlin data used.

**Outputs:**

- `outputs/phase_3/metadata/setup_decisions.json` (+ proximity_strategy)
- `outputs/phase_3/figures/exp_08b_proximity/*.png`

**Visualizations:**

| Experiment | Plot Description                 | Filename                            |
| ---------- | -------------------------------- | ----------------------------------- |
| exp_08b    | Proximity F1 comparison          | `proximity_f1_comparison.png`       |
| exp_08b    | Per-genus F1 (baseline/filtered) | `proximity_per_genus_f1.png`        |
| exp_08b    | Per-genus sample loss breakdown  | `proximity_sample_loss.png`         |

---

### 4.3 Exploratory Notebook: exp_08c_outlier_ablation.ipynb

**Purpose:** Determine outlier removal strategy using Phase 2 severity flags

Phase 2 computed three independent outlier detection methods (Z-Score, Mahalanobis, IQR) and combined them into summary columns:

- `outlier_severity`: none / low / medium / high (based on method agreement)
- `outlier_method_count`: 0-3 (number of methods flagging a tree as outlier)

These flags were deliberately kept as metadata (not acted upon) so that Phase 3 can make an informed, data-driven removal decision.

**Inputs:**

- `data/phase_2_splits/berlin_train.parquet` (or `_filtered`, based on exp_08b decision)
- `data/phase_2_splits/berlin_val.parquet` (or `_filtered`)
- `outputs/phase_3/metadata/setup_decisions.json` (chm_strategy, proximity_strategy)

**Key Tasks:**

1. **Outlier Distribution Analysis**
   - Count trees per severity level: none / low / medium / high
   - Per-genus breakdown: are some genera disproportionately flagged?
   - Visualize: outlier rate by genus (bar chart)

2. **Prepare Removal Variants**
   - `no_removal`: All trees (baseline)
   - `remove_high`: Drop trees with `outlier_severity == "high"` (flagged by ≥2 methods)
   - `remove_high_medium`: Drop trees with `outlier_severity in ["high", "medium"]`

3. **Cross-Validation Comparison**
   - Use Random Forest with default HP (stable baseline)
   - Apply CHM decision from exp_08 (consistent features)
   - 3-Fold Spatial Block CV on each variant
   - Record: F1, Per-Genus F1, Train-Val Gap

4. **Sample Loss vs. F1 Trade-off**
   - For each removal level: compute sample loss %
   - Plot: F1 vs. samples retained (trade-off curve)
   - Per-genus: check that no genus drops below minimum samples

5. **Decision**
   - Apply decision rules from config
   - Default to no removal if gains are marginal

**Decision Logic:**

```python
# Pseudocode
baseline_f1 = cv_f1(no_removal)
results = {}

for variant_name, remove_levels in [
    ("remove_high", ["high"]),
    ("remove_high_medium", ["high", "medium"]),
]:
    mask = ~train["outlier_severity"].isin(remove_levels)
    variant_f1 = cv_f1(train[mask])
    sample_loss = 1 - mask.sum() / len(train)
    results[variant_name] = {
        "f1": variant_f1,
        "sample_loss": sample_loss,
        "improvement": variant_f1 - baseline_f1
    }

# Select best variant that meets criteria
decision = "no_removal"  # Default
for name, r in sorted(results.items(), key=lambda x: x[1]["improvement"], reverse=True):
    if r["sample_loss"] > 0.15:
        continue  # Too much data loss
    if r["improvement"] < 0.02:
        continue  # Marginal gain
    decision = name
    break
```

**Note:** Same information-leak principle — only Berlin data used.

**Outputs:**

- `outputs/phase_3/metadata/setup_decisions.json` (+ outlier_strategy)
- `outputs/phase_3/figures/exp_08c_outlier/*.png`

**Visualizations:**

| Experiment | Plot Description                | Filename                            |
| ---------- | ------------------------------- | ----------------------------------- |
| exp_08c    | Outlier severity by genus       | `outlier_distribution_by_genus.png` |
| exp_08c    | F1 vs. samples retained curve   | `outlier_tradeoff_curve.png`        |
| exp_08c    | Per-genus F1 across variants    | `outlier_per_genus_f1.png`          |

---

### 4.4 Exploratory Notebook: exp_09_feature_reduction.ipynb

**Purpose:** Determine optimal feature count through importance-based selection

**Inputs:**

- `data/phase_2_splits/berlin_train.parquet` (or `_filtered`, based on exp_08b)
- `data/phase_2_splits/berlin_val.parquet` (or `_filtered`)
- `outputs/phase_3/metadata/setup_decisions.json` (chm_strategy, proximity_strategy, outlier_strategy)

**Key Tasks:**

1. **Compute Feature Importance**
   - Train RF with all features (applying CHM, proximity, and outlier decisions from exp_08/08b/08c)
   - Extract gain-based importance
   - Rank features

2. **Create Feature Subsets**
   - Top-30, Top-50, Top-80, All features

3. **Evaluate Each Subset**
   - 3-Fold Spatial Block CV with RF
   - Record F1, training time

4. **Pareto Analysis**
   - Plot: F1 vs. Feature Count
   - Identify knee point
   - Create literature comparison table (Hemmerling 2021: ~276 features, Immitzer 2019: 49 features)
   - Check for Hughes Effect (F1_all < F1_Top-k → feature curse documented)

5. **Decision Logging**
   - Select smallest k with F1 ≥ F1_all - 0.01
   - Save selected feature list
   - Document knee-point interpretation with literature context

**Outputs:**

- `outputs/phase_3/metadata/setup_decisions.json` (COMPLETE: chm_strategy, proximity_strategy, outlier_strategy, feature_set, selected_features, literature_comparison, knee_point_analysis)
- `outputs/phase_3/figures/exp_09_feature_reduction/*.png`

**Visualizations:**

| Experiment | Plot Description                      | Filename                                |
| ---------- | ------------------------------------- | --------------------------------------- |
| exp_09     | Feature importance ranking            | `feature_importance_ranking.png`        |
| exp_09     | Pareto curve (F1 vs. count)           | `pareto_curve.png`                      |
| exp_09     | Feature group contribution            | `feature_group_contribution.png`        |
| exp_09     | Pareto with literature context        | `feature_pareto_curve_literature.png`   |

---

## 5. JSON Schema

### 5.1 Setup Decisions Schema (`schemas/setup_decisions.schema.json`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Setup Decisions",
  "type": "object",
  "required": [
    "timestamp",
    "chm_strategy",
    "proximity_strategy",
    "outlier_strategy",
    "feature_set",
    "selected_features"
  ],
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
    "proximity_strategy": {
      "type": "object",
      "properties": {
        "decision": {
          "type": "string",
          "enum": ["baseline", "filtered"]
        },
        "reasoning": { "type": "string" },
        "sample_counts": {
          "type": "object",
          "properties": {
            "baseline_n": { "type": "integer" },
            "filtered_n": { "type": "integer" },
            "sample_loss_pct": { "type": "number" }
          }
        },
        "ablation_results": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "variant": { "type": "string" },
              "val_f1_mean": { "type": "number" },
              "val_f1_std": { "type": "number" },
              "n_samples": { "type": "integer" }
            }
          }
        }
      }
    },
    "outlier_strategy": {
      "type": "object",
      "properties": {
        "decision": {
          "type": "string",
          "enum": ["no_removal", "remove_high", "remove_high_medium"]
        },
        "reasoning": { "type": "string" },
        "sample_counts": {
          "type": "object",
          "properties": {
            "total_n": { "type": "integer" },
            "removed_n": { "type": "integer" },
            "sample_loss_pct": { "type": "number" }
          }
        },
        "ablation_results": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "variant": { "type": "string" },
              "val_f1_mean": { "type": "number" },
              "val_f1_std": { "type": "number" },
              "n_samples": { "type": "integer" },
              "sample_loss_pct": { "type": "number" }
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

---

## 6. Success Criteria

### 6.1 Functional Requirements

- [ ] All 4 exploratory notebooks execute without errors (exp_08, exp_08b, exp_08c, exp_09)
- [ ] `setup_decisions.json` created and validates against schema
- [ ] All 4 decisions documented with reasoning:
  - [ ] CHM strategy (from exp_08)
  - [ ] Proximity filter strategy (from exp_08b)
  - [ ] Outlier removal strategy (from exp_08c)
  - [ ] Feature set and selected features (from exp_09)
- [ ] All required figures generated (12 visualizations total)

### 6.2 Quality Requirements

- [ ] All experiments use only Berlin data (no information leakage)
- [ ] Decision rules from config applied consistently
- [ ] Pareto analysis identifies knee point
- [ ] Literature context documented (Hemmerling 2021, Immitzer 2019)
- [ ] All decisions reproducible from config + data

### 6.3 Documentation Requirements

- [ ] All decisions logged with quantitative rationale
- [ ] Trade-offs clearly documented (F1 vs. samples, F1 vs. feature count)
- [ ] Per-genus impacts analyzed where relevant

---

## 7. Dependencies and Outputs

### 7.1 Prerequisites

**Data:**
- `data/phase_2_splits/berlin_train.parquet`
- `data/phase_2_splits/berlin_val.parquet`
- `data/phase_2_splits/berlin_train_filtered.parquet`
- `data/phase_2_splits/berlin_val_filtered.parquet`

**Schemas:**
- `schemas/setup_decisions.schema.json`

**Config:**
- `configs/experiments/phase3_config.yaml`

### 7.2 Outputs

**Metadata:**
- `outputs/phase_3/metadata/setup_decisions.json`

**Figures:**
- `outputs/phase_3/figures/exp_08_chm_ablation/` (3 plots)
- `outputs/phase_3/figures/exp_08b_proximity/` (3 plots)
- `outputs/phase_3/figures/exp_08c_outlier/` (3 plots)
- `outputs/phase_3/figures/exp_09_feature_reduction/` (4 plots)

### 7.3 Consumed By

**Next PRD:** 003b (Berlin-Optimierung)
- Uses `setup_decisions.json` to configure datasets for algorithm comparison
- Uses selected feature set for all model training
- Uses CHM/proximity/outlier decisions to prepare data

---

## 8. Execution Order

The experiments must be run sequentially as each depends on the previous decision:

1. **exp_08_chm_ablation.ipynb** → Writes `chm_strategy` to setup_decisions.json
2. **exp_08b_proximity_ablation.ipynb** → Reads `chm_strategy`, writes `proximity_strategy`
3. **exp_08c_outlier_ablation.ipynb** → Reads `chm_strategy` + `proximity_strategy`, writes `outlier_strategy`
4. **exp_09_feature_reduction.ipynb** → Reads all prior decisions, writes `feature_set` + `selected_features` (COMPLETES setup_decisions.json)

**Estimated Runtime:** 6 hours total (3-fold CV on each experiment)

---

_Last Updated: 2026-02-07_
