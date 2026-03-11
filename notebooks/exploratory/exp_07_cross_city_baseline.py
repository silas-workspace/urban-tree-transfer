# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp_07: Cross-City Baseline Analysis
#
# **Phase 3 - Exploratory Analysis (Optional)**
#
# Descriptive analysis of domain shift between Berlin and Leipzig datasets **before** training. Generates hypotheses for transfer evaluation (03c) by quantifying:
# - Class distribution differences
# - Phenological profile divergence
# - Structural differences (CHM)
# - Feature distribution overlap
# - Statistical effect sizes (Cohen's d)
# - Correlation structure similarity
#
# **Note:** This notebook is **purely descriptive** — no training, no JSON outputs. Results inform interpretation of transfer experiments but are not used for decisions.

# %%
# ============================================================
# RUNTIME SETTINGS
# ============================================================
# Required: CPU (Standard)
# GPU: Not required
# High-RAM: Recommended (for correlation matrix on full feature set)
#
# SETUP: Add GITHUB_TOKEN to Colab Secrets (key icon in sidebar)
# ============================================================

import subprocess
from google.colab import userdata

# Get GitHub token from Colab Secrets (for private repo access)
token = userdata.get("GITHUB_TOKEN")
if not token:
    raise ValueError(
        "GITHUB_TOKEN not found in Colab Secrets.\n"
        "1. Click the key icon in the left sidebar\n"
        "2. Add a secret named 'GITHUB_TOKEN' with your GitHub token\n"
        "3. Toggle 'Notebook access' ON"
    )

# Install package from private GitHub repo
repo_url = f"git+https://{token}@github.com/SilasPignotti/urban-tree-transfer.git"
subprocess.run(["pip", "install", repo_url, "-q"], check=True)

print("OK: Package installed")

# %%
# Mount Google Drive for data files
from google.colab import drive

drive.mount("/content/drive")

print("Google Drive mounted")

# %%
# Package imports
from urban_tree_transfer.config import RANDOM_SEED
from urban_tree_transfer.experiments import (
    data_loading,
    evaluation,
    visualization,
)
from urban_tree_transfer.utils import ExecutionLog, save_figure, setup_plotting

from pathlib import Path
import warnings
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

setup_plotting()
log = ExecutionLog("exp_07_cross_city_baseline")

warnings.filterwarnings("ignore", category=UserWarning)

print("OK: Package imports complete")

# %%
# ============================================================
# CONFIGURATION
# ============================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
INPUT_DIR = DRIVE_DIR / "data" / "phase_2_splits"
OUTPUT_DIR = DRIVE_DIR / "data" / "phase_3_experiments"

LOGS_DIR = OUTPUT_DIR / "logs"
FIGURES_DIR = OUTPUT_DIR / "figures" / "exp_07_baseline"

for d in [LOGS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Input (Phase 2 Splits): {INPUT_DIR}")
print(f"Output (Phase 3):       {OUTPUT_DIR}")
print(f"Figures:                {FIGURES_DIR}")
print(f"Logs:                   {LOGS_DIR}")
print(f"Random seed:            {RANDOM_SEED}")

# %%
# ============================================================
# SECTION 1: Data Loading & Memory Optimization
# ============================================================

log.start_step("Data Loading")

# Load baseline splits (before proximity filtering)
berlin_train, berlin_val, berlin_test = data_loading.load_berlin_splits(
    INPUT_DIR, variant="baseline"
)
leipzig_finetune, leipzig_test = data_loading.load_leipzig_splits(
    INPUT_DIR, variant="baseline"
)

print(f"Berlin Train:      {len(berlin_train):,} samples")
print(f"Leipzig Finetune:  {len(leipzig_finetune):,} samples")

# Memory optimization: Convert float64 → float32
print("\nMemory optimization: Converting float64 → float32...")
for df in [berlin_train, leipzig_finetune]:
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
        
print(f"  Converted {len(float_cols)} float columns to float32")

log.end_step(status="success", records=len(berlin_train) + len(leipzig_finetune))

# %%
# ============================================================
# SECTION 2: Analysis 1 - Class Distribution Comparison
# ============================================================

log.start_step("Class Distribution Analysis")

berlin_genus_counts = berlin_train["genus_latin"].value_counts()
leipzig_genus_counts = leipzig_finetune["genus_latin"].value_counts()

# Get all unique genera (union)
all_genera = sorted(set(berlin_genus_counts.index) | set(leipzig_genus_counts.index))

# Align counts
berlin_aligned = berlin_genus_counts.reindex(all_genera, fill_value=0)
leipzig_aligned = leipzig_genus_counts.reindex(all_genera, fill_value=0)

# Create stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(all_genera))
width = 0.4

ax.bar(x - width/2, berlin_aligned.values, width=width, label="Berlin", color="#4C72B0")
ax.bar(x + width/2, leipzig_aligned.values, width=width, label="Leipzig", color="#55A868")

ax.set_xticks(x)
ax.set_xticklabels(all_genera, rotation=45, ha="right")
ax.set_ylabel("Sample Count")
ax.set_title("Genus Distribution: Berlin vs. Leipzig (Baseline Datasets)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

save_figure(fig, FIGURES_DIR / "genus_distribution_comparison.png")
print(f"Saved: {FIGURES_DIR / 'genus_distribution_comparison.png'}")

# Print top-5 genera for each city
print("\nTop-5 Genera in Berlin:")
print(berlin_genus_counts.head(5).to_string())
print("\nTop-5 Genera in Leipzig:")
print(leipzig_genus_counts.head(5).to_string())

log.end_step(status="success", records=len(all_genera))

# %%
# ============================================================
# SECTION 3: Analysis 2 - Phenological Profiles (Top-5 Genera)
# ============================================================

log.start_step("Phenological Profile Analysis")

# Get top-5 genera by total counts across both cities
total_counts = berlin_genus_counts.add(leipzig_genus_counts, fill_value=0)
top5_genera = total_counts.nlargest(5).index.tolist()

print(f"Analyzing top-5 genera: {top5_genera}")

# Extract NDVI columns (monthly NDVI features)
import re

ndvi_pattern = re.compile(r"^NDVI_(\d{2})(?:_mean)?$")
ndvi_cols = [col for col in berlin_train.columns if ndvi_pattern.match(col)]
if not ndvi_cols:
    ndvi_cols = [
        col
        for col in berlin_train.columns
        if col.startswith("NDVI_") and not col.startswith("NDVIre_")
    ]

def _month_from_col(col: str) -> int | None:
    match = ndvi_pattern.match(col)
    if match:
        return int(match.group(1))
    try:
        return int(col.rsplit("_", 1)[1])
    except ValueError:
        return None

ndvi_cols = sorted(
    ndvi_cols,
    key=lambda col: (
        _month_from_col(col) is None,
        _month_from_col(col) or 0,
        col,
    ),
)

if len(ndvi_cols) == 0:
    print("⚠️  Warning: No NDVI columns found. Skipping phenological analysis.")
else:
    months = [m for m in (_month_from_col(col) for col in ndvi_cols) if m is not None]
    if len(months) != len(ndvi_cols):
        months = list(range(1, len(ndvi_cols) + 1))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, genus in enumerate(top5_genera):
        ax = axes[idx]
        
        # Berlin profile
        berlin_genus = berlin_train[berlin_train["genus_latin"] == genus]
        if len(berlin_genus) > 0:
            berlin_ndvi_mean = berlin_genus[ndvi_cols].mean(axis=0).values
            berlin_ndvi_std = berlin_genus[ndvi_cols].std(axis=0).values
            ax.plot(months, berlin_ndvi_mean, marker="o", label="Berlin", color="#4C72B0")
            ax.fill_between(
                months,
                berlin_ndvi_mean - berlin_ndvi_std,
                berlin_ndvi_mean + berlin_ndvi_std,
                alpha=0.2,
                color="#4C72B0",
            )
        
        # Leipzig profile
        leipzig_genus = leipzig_finetune[leipzig_finetune["genus_latin"] == genus]
        if len(leipzig_genus) > 0:
            leipzig_ndvi_mean = leipzig_genus[ndvi_cols].mean(axis=0).values
            leipzig_ndvi_std = leipzig_genus[ndvi_cols].std(axis=0).values
            ax.plot(months, leipzig_ndvi_mean, marker="s", label="Leipzig", color="#55A868")
            ax.fill_between(
                months,
                leipzig_ndvi_mean - leipzig_ndvi_std,
                leipzig_ndvi_mean + leipzig_ndvi_std,
                alpha=0.2,
                color="#55A868",
            )
        
        ax.set_xlabel("Month")
        ax.set_ylabel("NDVI (mean ± std)")
        ax.set_title(f"{genus}")
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Hide unused subplot
    if len(top5_genera) < 6:
        axes[5].axis("off")
    
    plt.suptitle("Phenological Profiles: Top-5 Genera (Berlin vs. Leipzig)", fontsize=14)
    plt.tight_layout()
    save_figure(fig, FIGURES_DIR / "phenological_profiles_top5.png")
    print(f"Saved: {FIGURES_DIR / 'phenological_profiles_top5.png'}")

log.end_step(status="success")

# %%
# ============================================================
# SECTION 4: Analysis 3 - CHM Distribution per Genus
# ============================================================

log.start_step("CHM Distribution Analysis")

# Use CHM_1m (raw CHM mean)
if "CHM_1m" not in berlin_train.columns:
    print("⚠️  Warning: CHM_1m column not found. Skipping CHM analysis.")
else:
    # Prepare data for violin plots
    berlin_chm = berlin_train[["genus_latin", "CHM_1m"]].copy()
    berlin_chm["city"] = "Berlin"
    leipzig_chm = leipzig_finetune[["genus_latin", "CHM_1m"]].copy()
    leipzig_chm["city"] = "Leipzig"
    
    combined_chm = pd.concat([berlin_chm, leipzig_chm], ignore_index=True)
    
    # Filter to top-5 genera for clarity
    combined_chm = combined_chm[combined_chm["genus_latin"].isin(top5_genera)]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(
        data=combined_chm,
        x="genus_latin",
        y="CHM_1m",
        hue="city",
        split=True,
        inner="quartile",
        palette={"Berlin": "#4C72B0", "Leipzig": "#55A868"},
        ax=ax,
    )
    
    ax.set_xlabel("Genus")
    ax.set_ylabel("CHM (Canopy Height Model) [m]")
    ax.set_title("CHM Distribution per Genus: Berlin vs. Leipzig")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    
    save_figure(fig, FIGURES_DIR / "chm_violin_per_genus.png")
    print(f"Saved: {FIGURES_DIR / 'chm_violin_per_genus.png'}")

log.end_step(status="success")

# %%
# ============================================================
# SECTION 5: Analysis 4 - Feature Distribution Overlap
# ============================================================

log.start_step("Feature Distribution Overlap")

# Get top-20 most important features (use heuristic: NDVI, EVI, SWIR, CHM)
# Since we don't have feature importance yet, select based on common patterns
feature_cols = data_loading.get_feature_columns(
    berlin_train, include_chm=True, chm_features=["CHM_1m"]
)

# Select top-20: Mix of spectral indices and CHM
priority_patterns = ["NDVI", "EVI", "SWIR", "NIR", "RED", "CHM"]
top20_features = []
for pattern in priority_patterns:
    matches = [f for f in feature_cols if pattern in f and f not in top20_features]
    top20_features.extend(matches[:4])  # Take up to 4 per pattern
    if len(top20_features) >= 20:
        break

top20_features = top20_features[:20]
print(f"Selected {len(top20_features)} features for distribution analysis")

# Create ridge plot (overlapping KDE curves)
fig, axes = plt.subplots(len(top20_features), 1, figsize=(10, len(top20_features) * 0.8))
fig.subplots_adjust(hspace=0.5)

for idx, feature in enumerate(top20_features):
    ax = axes[idx] if len(top20_features) > 1 else axes
    
    # KDE for Berlin
    berlin_values = berlin_train[feature].dropna()
    if len(berlin_values) > 10:
        berlin_values.plot.kde(ax=ax, color="#4C72B0", label="Berlin", linewidth=2)
    
    # KDE for Leipzig
    leipzig_values = leipzig_finetune[feature].dropna()
    if len(leipzig_values) > 10:
        leipzig_values.plot.kde(ax=ax, color="#55A868", label="Leipzig", linewidth=2)
    
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title(feature, fontsize=9, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    if idx == 0:
        ax.legend(loc="upper right")
    else:
        ax.legend().set_visible(False)

plt.suptitle("Feature Distribution Overlap: Berlin vs. Leipzig (Top-20 Features)", fontsize=12, y=0.995)
save_figure(fig, FIGURES_DIR / "feature_distribution_overlap.png")
print(f"Saved: {FIGURES_DIR / 'feature_distribution_overlap.png'}")

log.end_step(status="success", records=len(top20_features))

# %%
# ============================================================
# SECTION 6: Analysis 5 - Cohen's d Heatmap
# ============================================================

log.start_step("Cohen's d Effect Size Analysis")

# Compute Cohen's d for top-5 genera × top-20 features
cohens_d_matrix = []

for genus in top5_genera:
    row = []
    berlin_genus = berlin_train[berlin_train["genus_latin"] == genus]
    leipzig_genus = leipzig_finetune[leipzig_finetune["genus_latin"] == genus]
    
    for feature in top20_features:
        berlin_vals = berlin_genus[feature].dropna().values
        leipzig_vals = leipzig_genus[feature].dropna().values
        
        if len(berlin_vals) >= 2 and len(leipzig_vals) >= 2:
            try:
                d = evaluation.compute_cohens_d(berlin_vals, leipzig_vals)
                row.append(d)
            except ValueError:
                row.append(np.nan)
        else:
            row.append(np.nan)
    
    cohens_d_matrix.append(row)

cohens_d_df = pd.DataFrame(
    cohens_d_matrix,
    index=top5_genera,
    columns=[f.replace("_mean", "").replace("_2023", "")[:15] for f in top20_features],
)

# Heatmap
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    cohens_d_df,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1.5,
    vmax=1.5,
    cbar_kws={"label": "Cohen's d"},
    ax=ax,
)

ax.set_xlabel("Feature")
ax.set_ylabel("Genus")
ax.set_title("Cohen's d Effect Sizes: Berlin vs. Leipzig (Top-5 Genera × Top-20 Features)")

save_figure(fig, FIGURES_DIR / "cohens_d_heatmap.png")
print(f"Saved: {FIGURES_DIR / 'cohens_d_heatmap.png'}")

# Interpretation guide
print("\nCohen's d Interpretation:")
print("  |d| < 0.2:  Negligible (Transfer should work well)")
print("  0.2-0.5:    Small (Moderate transfer gap expected)")
print("  0.5-0.8:    Medium (Noticeable transfer gap)")
print("  |d| > 0.8:  Large (Significant transfer gap, genus may not transfer well)")

# Summary statistics
abs_d = np.abs(cohens_d_df.values[~np.isnan(cohens_d_df.values)])
print(f"\nSummary:")
print(f"  Mean |d|:    {abs_d.mean():.3f}")
print(f"  Median |d|:  {np.median(abs_d):.3f}")
print(f"  Max |d|:     {abs_d.max():.3f}")

log.end_step(status="success")

# %%
# ============================================================
# SECTION 7: Analysis 6 - Correlation Structure Comparison
# ============================================================

log.start_step("Correlation Structure Analysis")

# Compute correlation matrices for top-20 features
berlin_corr = berlin_train[top20_features].corr()
leipzig_corr = leipzig_finetune[top20_features].corr()

# Side-by-side heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Berlin correlation
sns.heatmap(
    berlin_corr,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    cbar_kws={"label": "Pearson r"},
    ax=ax1,
    xticklabels=[f[:10] for f in top20_features],
    yticklabels=[f[:10] for f in top20_features],
)
ax1.set_title("Berlin: Feature Correlation Structure")

# Leipzig correlation
sns.heatmap(
    leipzig_corr,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    cbar_kws={"label": "Pearson r"},
    ax=ax2,
    xticklabels=[f[:10] for f in top20_features],
    yticklabels=[f[:10] for f in top20_features],
)
ax2.set_title("Leipzig: Feature Correlation Structure")

plt.suptitle("Correlation Structure Comparison: Berlin vs. Leipzig", fontsize=14)
plt.tight_layout()
save_figure(fig, FIGURES_DIR / "correlation_structure_comparison.png")
print(f"Saved: {FIGURES_DIR / 'correlation_structure_comparison.png'}")

# Compute correlation between correlation matrices (structure similarity)
# Flatten upper triangles (exclude diagonal)
mask = np.triu(np.ones_like(berlin_corr, dtype=bool), k=1)
berlin_flat = berlin_corr.values[mask]
leipzig_flat = leipzig_corr.values[mask]

struct_corr, _ = pearsonr(berlin_flat, leipzig_flat)
print(f"\nCorrelation structure similarity (Pearson r): {struct_corr:.3f}")
print("  r > 0.9: Very similar correlation structures (models should transfer well)")
print("  r < 0.7: Different correlation structures (NN models may struggle more than ML)")

log.end_step(status="success")

# %%
# ============================================================
# SECTION 8: Summary & Next Steps
# ============================================================

# Save execution log
log.summary()
log_path = LOGS_DIR / f"{log.notebook}_execution.json"
log.save(log_path)
print(f"Execution log saved: {log_path}")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE: exp_07 Cross-City Baseline Analysis")
print("=" * 70)
print(f"\nAnalyses completed:")
print(f"  1. Class Distribution Comparison")
print(f"  2. Phenological Profiles (Top-5 genera)")
print(f"  3. CHM Distribution per Genus")
print(f"  4. Feature Distribution Overlap (Top-20 features)")
print(f"  5. Cohen's d Effect Sizes (Statistical domain shift)")
print(f"  6. Correlation Structure Comparison")
print(f"\nOutputs:")
print(f"  - Figures: {FIGURES_DIR}")
print(f"  - Logs: {log_path}")
print(f"\nInterpretation:")
print(f"  Use these descriptive analyses to:")
print(f"  - Generate hypotheses for transfer evaluation (03c)")
print(f"  - Explain per-genus transfer robustness differences")
print(f"  - Validate literature claims (e.g., conifer spectral stability)")
print(f"\nNext Steps:")
print(f"  - Review all visualizations in {FIGURES_DIR}")
print(f"  - Document key findings in methodology docs")
print(f"  - Continue with critical path: exp_08 → exp_08b → exp_08c → exp_09")
print("=" * 70)
