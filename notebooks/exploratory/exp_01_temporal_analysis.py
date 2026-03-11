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
# # Exploratory Notebook: Temporal Feature Selection (exp_01)
#
# **Goal:** Analyze monthly Sentinel-2 features using Jeffries-Matusita (JM) distance to identify months with strong genus discrimination.
#
# **Outputs:**
# - `outputs/phase_2/metadata/temporal_selection.json` (on Google Drive)
# - Publication-quality plots saved to `outputs/phase_2/figures/exp_01_temporal`
#

# %%
# ============================================================
# RUNTIME SETTINGS
# ============================================================
# Required: CPU (Standard)
# GPU: Not required
# High-RAM: Recommended for large datasets
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
from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.utils import ExecutionLog, save_figure, setup_plotting
from urban_tree_transfer.utils.plotting import PUBLICATION_STYLE

from itertools import combinations
from pathlib import Path
from datetime import datetime, timezone
import json
import re

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

setup_plotting()
log = ExecutionLog("exp_01_temporal_analysis")

print("OK: Package imports complete")


# %%
# ============================================================
# CONFIGURATION
# ============================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
INPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
OUTPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
METADATA_DIR = OUTPUT_DIR / "metadata"
LOGS_DIR = OUTPUT_DIR / "logs"
PLOTS_DIR = OUTPUT_DIR / "figures" / "exp_01_temporal"

CITIES = ["berlin", "leipzig"]
MONTHS = list(range(1, 13))

MIN_SAMPLES_PER_GENUS = 500
MAX_SAMPLES_PER_GENUS = 10000  # Balance: 10k gives stable JM estimates, ~80% faster than full dataset
RANDOM_SEED = 42

for d in [METADATA_DIR, LOGS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Input:  {INPUT_DIR}")
print(f"Output: {METADATA_DIR}")
print(f"Plots:  {PLOTS_DIR}")
print(f"Cities: {CITIES}")

# %% [markdown]
# ## Data Loading & Validation
#
# Load Phase 2a feature data for Berlin and Leipzig and validate expected schema and temporal feature layout.

# %%
log.start_step("Data Loading & Validation")

try:
    city_data = {}
    all_feature_cols = None
    feature_cols_by_month = None
    s2_feature_bases = None

    for city in CITIES:
        path = INPUT_DIR / f"trees_with_features_{city}.gpkg"
        print(f"Loading: {path}")
        df = gpd.read_file(path)

        # Normalize city column to lowercase for consistency
        if "city" in df.columns:
            df["city"] = df["city"].astype(str).str.strip().str.lower()

        # Basic validation
        required_cols = ["tree_id", "city", "genus_latin", "geometry"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Identify S2 feature columns
        pattern = re.compile(r"^(.+)_([0-1][0-9])$")
        feature_cols = []
        feature_map = {}
        for col in df.columns:
            match = pattern.match(col)
            if match:
                base, month = match.group(1), int(match.group(2))
                if 1 <= month <= 12:
                    feature_cols.append(col)
                    feature_map.setdefault(month, []).append(col)

        if not feature_cols:
            raise ValueError("No Sentinel-2 feature columns found.")

        for month in MONTHS:
            if month not in feature_map:
                raise ValueError(f"Missing features for month {month:02d}")

        # Consistency checks across cities
        if all_feature_cols is None:
            all_feature_cols = sorted(feature_cols)
            feature_cols_by_month = {m: sorted(cols) for m, cols in feature_map.items()}
            s2_feature_bases = sorted({pattern.match(c).group(1) for c in all_feature_cols})
        else:
            if sorted(feature_cols) != all_feature_cols:
                raise ValueError(f"Feature columns mismatch between cities: {city}")

        # Filter to genera with sufficient samples
        genus_counts = df["genus_latin"].value_counts()
        viable_genera = genus_counts[genus_counts >= MIN_SAMPLES_PER_GENUS].index.tolist()
        df = df[df["genus_latin"].isin(viable_genera)].copy()

        # Optional sampling per genus to reduce compute time
        if MAX_SAMPLES_PER_GENUS is not None:
            sampled = []
            for genus in viable_genera:
                subset = df[df["genus_latin"] == genus]
                if len(subset) > MAX_SAMPLES_PER_GENUS:
                    subset = subset.sample(MAX_SAMPLES_PER_GENUS, random_state=RANDOM_SEED)
                sampled.append(subset)
            df = gpd.GeoDataFrame(pd.concat(sampled, ignore_index=True), crs=df.crs)

        city_data[city] = df
        print(f"  {city}: {len(df):,} trees, {len(viable_genera)} viable genera")

    log.end_step(status="success", records=sum(len(v) for v in city_data.values()))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## JM Distance Computation
#
# Compute Jeffries-Matusita distance per month, per city, and per genus pair. JM is computed per feature and averaged across all 23 Sentinel-2 features for each month.

# %%
def compute_jm_distance(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    """Compute Jeffries-Matusita distance (standard formula)."""
    eps = 1e-6
    sigma1 = max(float(sigma1), eps)
    sigma2 = max(float(sigma2), eps)
    mean_diff = float(mu1) - float(mu2)

    sigma_avg = 0.5 * (sigma1 + sigma2)
    term1 = (1.0 / 8.0) * (mean_diff ** 2) / sigma_avg
    term2 = 0.5 * np.log(sigma_avg / np.sqrt(sigma1 * sigma2))
    b_distance = term1 + term2

    jm = 2.0 * (1.0 - np.exp(-b_distance))
    jm = float(np.clip(jm, 0.0, 2.0))
    return jm

def compute_feature_stats(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby("genus_latin")[feature_cols]
    means = grouped.mean()
    stds = grouped.std(ddof=1)
    counts = grouped.count()
    return means, stds, counts

log.start_step("JM Distance Computation")

try:
    jm_records = []
    viable_genera_by_city = {}

    for city, df in city_data.items():
        genus_counts = df["genus_latin"].value_counts()
        viable_genera = genus_counts[genus_counts >= MIN_SAMPLES_PER_GENUS].index.tolist()
        viable_genera_by_city[city] = sorted(viable_genera)

        print(f"Computing stats for {city}...")
        means, stds, counts = compute_feature_stats(df, all_feature_cols)

        genus_pairs = list(combinations(viable_genera, 2))
        print(f"  Genus pairs: {len(genus_pairs)}")

        for month in MONTHS:
            month_cols = feature_cols_by_month[month]

            for genus_a, genus_b in genus_pairs:
                jm_values = []
                for col in month_cols:
                    if counts.loc[genus_a, col] < MIN_SAMPLES_PER_GENUS:
                        continue
                    if counts.loc[genus_b, col] < MIN_SAMPLES_PER_GENUS:
                        continue

                    mu1 = means.loc[genus_a, col]
                    mu2 = means.loc[genus_b, col]
                    sigma1 = stds.loc[genus_a, col]
                    sigma2 = stds.loc[genus_b, col]

                    if not np.isfinite(mu1) or not np.isfinite(mu2):
                        continue
                    if not np.isfinite(sigma1) or not np.isfinite(sigma2):
                        continue

                    jm = compute_jm_distance(mu1, sigma1, mu2, sigma2)
                    if np.isfinite(jm):
                        jm_values.append(jm)

                if jm_values:
                    jm_mean = float(np.mean(jm_values))
                    jm_records.append(
                        {
                            "city": city,
                            "month": month,
                            "genus_pair": f"{genus_a} vs {genus_b}",
                            "jm_distance": jm_mean,
                            "features_used": len(jm_values),
                        }
                    )

    jm_df = pd.DataFrame(jm_records)
    if jm_df.empty:
        raise ValueError("JM computation resulted in empty output. Check data and thresholds.")

    # Sanity checks
    if jm_df["jm_distance"].min() < 0 or jm_df["jm_distance"].max() > 2.0:
        raise ValueError("JM distances out of expected range [0, 2].")
    if jm_df["jm_distance"].isna().any():
        raise ValueError("NaN values detected in JM distances.")

    log.end_step(status="success", records=len(jm_df))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Visualization
#
# Generate publication-quality plots for temporal JM trends and genus-pair heatmaps.

# %%
# Plot 1: JM Distance Line Chart (Berlin vs Leipzig)
plot_df = (
    jm_df.groupby(["city", "month"])["jm_distance"]
    .agg(["mean", "std"])
    .reset_index()
)

city_colors = {
    "berlin": "#1f77b4",
    "leipzig": "#ff7f0e",
}

fig, ax = plt.subplots(figsize=(12, 6))
for city in CITIES:
    subset = plot_df[plot_df["city"] == city]
    ax.errorbar(
        subset["month"],
        subset["mean"],
        yerr=subset["std"],
        label=city.title(),
        color=city_colors.get(city),
        marker="o",
        linewidth=2,
        capsize=3,
    )

ax.set_title("Temporal Separability (JM Distance) by Month", fontsize=14)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Mean JM Distance", fontsize=12)
ax.set_xticks(MONTHS)
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=10)

save_figure(fig, PLOTS_DIR / "jm_distance_by_month.png", dpi=300)
print(f"Saved: {PLOTS_DIR / 'jm_distance_by_month.png'}")


# %%
# Plot 2: JM Distance Heatmaps (top genus pairs)
TOP_GENUS_COUNT = 6  # 6 genera -> 15 pairs

for city in CITIES:
    df_city = city_data[city]
    top_genera = df_city["genus_latin"].value_counts().head(TOP_GENUS_COUNT).index.tolist()
    top_pairs = [f"{a} vs {b}" for a, b in combinations(top_genera, 2)]

    heatmap_df = jm_df[(jm_df["city"] == city) & (jm_df["genus_pair"].isin(top_pairs))]
    if heatmap_df.empty:
        print(f"No heatmap data for {city}.")
        continue

    pivot = heatmap_df.pivot(index="genus_pair", columns="month", values="jm_distance")
    pivot = pivot.reindex(index=top_pairs)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "JM Distance"},
        ax=ax,
    )

    ax.set_title(
        f"Genus Separability Across Months ({city.title()})",
        fontsize=14,
    )
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Genus Pair", fontsize=12)
    ax.set_xticklabels([f"{m:02d}" for m in MONTHS], rotation=0)

    filename = f"jm_heatmap_{city}.png"
    save_figure(fig, PLOTS_DIR / filename, dpi=300)
    print(f"Saved: {PLOTS_DIR / filename}")


# %% [markdown]
# ## Cross-City JM Consistency Analysis
#
# Validate that high-JM months are consistent across Berlin and Leipzig. This ensures selected months are transfer-robust, not city-specific.
#
# **Method:** Spearman rank correlation between city-specific JM rankings
#
# **Threshold:** ρ > 0.7 indicates high consistency (average JM selection valid)
#

# %%
import scipy.stats

log.start_step("Cross-City JM Consistency Analysis")

try:
    # Compute monthly JM means per city (city column already normalized at data load)
    jm_by_city_month = (
        jm_df.groupby(["city", "month"])["jm_distance"]
        .mean()
        .reset_index()
    )

    jm_berlin = (
        jm_by_city_month[jm_by_city_month["city"] == "berlin"]
        .set_index("month")["jm_distance"]
    )
    jm_leipzig = (
        jm_by_city_month[jm_by_city_month["city"] == "leipzig"]
        .set_index("month")["jm_distance"]
    )

    # Ensure both have all 12 months
    jm_berlin = jm_berlin.reindex(MONTHS)
    jm_leipzig = jm_leipzig.reindex(MONTHS)

    nan_months_berlin = jm_berlin[jm_berlin.isna()].index.tolist()
    nan_months_leipzig = jm_leipzig[jm_leipzig.isna()].index.tolist()
    if nan_months_berlin or nan_months_leipzig:
        print(
            "Warning: NaN monthly JM means detected. "
            "These months will be ranked as worst."
        )
        if nan_months_berlin:
            print(f"  Berlin NaN months: {nan_months_berlin}")
        if nan_months_leipzig:
            print(f"  Leipzig NaN months: {nan_months_leipzig}")

    # Rank months (1 = best, 12 = worst)
    jm_berlin = jm_berlin.fillna(-float("inf"))
    jm_leipzig = jm_leipzig.fillna(-float("inf"))
    rank_berlin = jm_berlin.rank(ascending=False, method="average").astype(int)
    rank_leipzig = jm_leipzig.rank(ascending=False, method="average").astype(int)

    # Spearman rank correlation
    rho, p_value = scipy.stats.spearmanr(rank_berlin, rank_leipzig)

    print(f"Spearman ρ: {rho:.3f} (p={p_value:.4f})")

    if p_value < 0.05:
        if rho > 0.7:
            consistency_interpretation = "high consistency - average JM selection valid"
            status_icon = "✅"
        elif rho > 0.5:
            consistency_interpretation = "moderate consistency - use conservative intersection"
            status_icon = "⚠️"
        else:
            consistency_interpretation = "low consistency - city-specific months detected"
            status_icon = "❌"
    else:
        consistency_interpretation = "correlation not significant - proceed with caution"
        status_icon = "⚠️"

    print(f"{status_icon} {consistency_interpretation}")

    # Identify top-8 months per city
    top8_berlin = jm_berlin.nlargest(8).index.tolist()
    top8_leipzig = jm_leipzig.nlargest(8).index.tolist()

    # Find consistent months (intersection)
    consistent_months = sorted(set(top8_berlin) & set(top8_leipzig))
    berlin_specific = sorted(set(top8_berlin) - set(consistent_months))
    leipzig_specific = sorted(set(top8_leipzig) - set(consistent_months))

    consistency_data = {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "interpretation": consistency_interpretation,
        "top8_berlin": top8_berlin,
        "top8_leipzig": top8_leipzig,
        "consistent_months": consistent_months,
        "berlin_only": berlin_specific,
        "leipzig_only": leipzig_specific,
    }

    log.end_step(status="success", records=len(MONTHS))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# Plot: Rank consistency scatter plot
fig, ax = plt.subplots(figsize=(10, 10))

# Scatter plot with month labels
for month in MONTHS:
    ax.scatter(
        rank_berlin[month],
        rank_leipzig[month],
        s=200,
        alpha=0.6,
        color="steelblue",
        edgecolors="black",
        linewidth=1.5,
    )
    ax.text(
        rank_berlin[month] + 0.15,
        rank_leipzig[month] + 0.15,
        str(month),
        fontsize=11,
        ha="left",
    )

# Diagonal line (perfect consistency)
ax.plot([1, 12], [1, 12], "r--", linewidth=2, alpha=0.5, label="Perfect Consistency")

# Formatting
ax.set_xlabel("Rank in Berlin (1=best, 12=worst)", fontsize=12)
ax.set_ylabel("Rank in Leipzig (1=best, 12=worst)", fontsize=12)
ax.set_title(
    f"Cross-City JM Rank Consistency\nSpearman ρ = {rho:.3f} (p={p_value:.4f})",
    fontsize=14,
)
ax.set_xlim(0.5, 12.5)
ax.set_ylim(0.5, 12.5)
ax.set_xticks(range(1, 13))
ax.set_yticks(range(1, 13))
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=10)
ax.set_aspect("equal")

save_figure(fig, PLOTS_DIR / "jm_rank_consistency.png", dpi=300)
print(f"Saved: {PLOTS_DIR / 'jm_rank_consistency.png'}")


# %% [markdown]
# ## Month Selection Logic
#
# Select months using a top-N approach with seasonal balance to ensure phenological coverage.

# %%
monthly_jm = jm_df.groupby("month")["jm_distance"].mean().sort_values(ascending=False)
threshold = float(monthly_jm.quantile(0.75))

def select_months_with_balance(monthly_series: pd.Series, top_n: int = 8) -> list[int]:
    """Select months with seasonal balance (unchanged from original)."""
    seasons = {
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
        "winter": [12, 1, 2],
    }
    min_per_season = {
        "spring": 1,
        "summer": 2,
        "autumn": 1,
        "winter": 0,
    }

    selected = list(monthly_series.head(top_n).index)

    # Ensure seasonal balance
    for season, months in seasons.items():
        needed = min_per_season[season]
        current = [m for m in selected if m in months]
        if len(current) >= needed:
            continue
        candidates = [m for m in months if m not in selected]
        candidates = sorted(candidates, key=lambda m: monthly_series.get(m, -1), reverse=True)
        for m in candidates:
            if len(current) >= needed:
                break
            selected.append(m)
            current.append(m)

    # Cap size to 10 if seasonal additions push over
    if len(selected) > 10:
        selected = sorted(selected, key=lambda m: monthly_series.get(m, -1), reverse=True)[:10]

    selected = sorted(set(selected))
    return selected

# Consistency-aware selection
if rho > 0.7:
    # High consistency: use average JM (current approach)
    selected_months = select_months_with_balance(monthly_jm, top_n=8)
    selection_method_used = "average_jm_with_consistency_validation"
    print(f"✅ High consistency (ρ={rho:.3f}) - using average JM selection")
else:
    # Low consistency: use conservative intersection
    selected_months = list(consistent_months)
    selection_method_used = "consistent_intersection_only"
    print(f"⚠️ Low consistency (ρ={rho:.3f}) - using consistent months only")

    # Apply seasonal balance to intersection if needed
    if len(selected_months) < 6:
        print(
            f"⚠️ Only {len(selected_months)} consistent months - adding top-ranked months to reach 6"
        )
        additional_needed = 6 - len(selected_months)
        candidates = [m for m in monthly_jm.index if m not in selected_months]
        selected_months = sorted(selected_months + candidates[:additional_needed])

selected_months = sorted(set(selected_months))
rejected_months = [m for m in MONTHS if m not in selected_months]

print(f"Selected months: {selected_months}")
print(f"Rejected months: {rejected_months}")


# %% [markdown]
# ## JSON Output
#
# Save selected months and JM statistics for downstream use in Phase 2b.

# %%
monthly_stats = (
    jm_df.groupby("month")["jm_distance"]
    .agg(["mean", "std", "min", "max"])
    .round(4)
)

viable_genera_all = sorted(set.intersection(*[set(v) for v in viable_genera_by_city.values()]))

output = {
    "analysis_date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "input_files": [
        "trees_with_features_berlin.gpkg",
        "trees_with_features_leipzig.gpkg",
    ],
    "total_trees_analyzed": int(sum(len(df) for df in city_data.values())),
    "viable_genera": viable_genera_all,
    "genus_pairs_analyzed": int(jm_df["genus_pair"].nunique()),
    "cross_city_validation": consistency_data,
    "monthly_jm_statistics": {
        str(month): {
            "mean": float(monthly_stats.loc[month, "mean"]),
            "std": float(monthly_stats.loc[month, "std"]),
            "min": float(monthly_stats.loc[month, "min"]),
            "max": float(monthly_stats.loc[month, "max"]),
        }
        for month in MONTHS
    },
    "selection_method": selection_method_used,
    "selection_threshold": float(threshold),
    "selected_months": selected_months,
    "rejected_months": rejected_months,
    "rationale": (
        f"Selected {len(selected_months)} months using {selection_method_used}. "
        f"Cross-city rank correlation ρ={rho:.3f} indicates "
        f"{'high' if rho > 0.7 else 'moderate'} consistency. "
        "Selected months are discriminative in both Berlin and Leipzig."
    ),
}

output_path = METADATA_DIR / "temporal_selection.json"
output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"Saved JSON: {output_path}")


# %% [markdown]
# ## Summary & Manual Sync Instructions

# %%
print("=" * 60)
print("TEMPORAL SELECTION SUMMARY")
print("=" * 60)
print("\n--- CROSS-CITY CONSISTENCY ---")
print(f"Spearman ρ: {rho:.3f} (p={p_value:.4f})")
print(f"Interpretation: {consistency_interpretation}")
print(f"Top-8 overlap: {len(consistent_months)}/8 months")
if berlin_specific or leipzig_specific:
    print("City-specific months:")
    if berlin_specific:
        print(f"  Berlin-only: {berlin_specific}")
    if leipzig_specific:
        print(f"  Leipzig-only: {leipzig_specific}")
print(f"Selection method: {selection_method_used}")
print("")
print("Total months analyzed: 12")
print(f"Months selected: {len(selected_months)}")
print(f"Selected: {selected_months}")
print(f"Mean JM (selected): {monthly_jm[selected_months].mean():.3f}")
selected_mean = monthly_jm[selected_months].mean()
if selected_mean < 0.7:
    interpretation = "low separability"
elif selected_mean < 1.0:
    interpretation = "moderate separability"
else:
    interpretation = "good separability"
print(f"Interpretation: {interpretation} (JM range: 0=overlap, 1=acceptable, 2=perfect)")
print(f"Mean JM (rejected): {monthly_jm[rejected_months].mean():.3f}")
print("\nSeasonal Coverage:")
print(f"  Spring (3-5):  {len([m for m in selected_months if 3 <= m <= 5])}/3")
print(f"  Summer (6-8):  {len([m for m in selected_months if 6 <= m <= 8])}/3")
print(f"  Autumn (9-11): {len([m for m in selected_months if 9 <= m <= 11])}/3")
print(f"  Winter (12,1,2): {len([m for m in selected_months if m in [12, 1, 2]])}/3")
print("\nJM Distance Quality:")
print(f"  Selected months mean: {monthly_jm[selected_months].mean():.3f}")
print(f"  Rejected months mean: {monthly_jm[rejected_months].mean():.3f}")
print(f"  Overall range: [{jm_df['jm_distance'].min():.3f}, {jm_df['jm_distance'].max():.3f}]")
print("\nNote: Legacy JM values were typically 0.5-1.2. Values are used for")
print("      relative comparison between months, not absolute thresholds.")

# Save execution log
log.summary()
log_path = LOGS_DIR / f"{log.notebook}_execution.json"
log.save(log_path)
print(f"Execution log saved: {log_path}")

print("\n--- JSON OUTPUT ---")
print(f"  {output_path.name}")

print("\n--- PLOTS CREATED ---")
for f in sorted(PLOTS_DIR.glob("*.png")):
    print(f"  {f.name}")


# %% [markdown]
# ## ⚠️ Manual Sync Required
#
# This notebook generated `temporal_selection.json` on Google Drive.
#
# **Next Steps:**
# 1. Download `temporal_selection.json` from Drive to local repo
# 2. Place in: `outputs/phase_2/metadata/temporal_selection.json`
# 3. Commit to git
# 4. Push to GitHub
# 5. Continue with Task 6 (Quality Control implementation)
#
# **File Path on Drive:**
# `/content/drive/MyDrive/dev/urban-tree-transfer/outputs/phase_2/metadata/temporal_selection.json`
#
