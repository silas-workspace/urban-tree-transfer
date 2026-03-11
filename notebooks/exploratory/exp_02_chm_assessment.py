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
# # Exploratory Notebook: CHM Assessment (exp_02)
#
# **Goal:** Evaluate CHM feature quality, determine plant year threshold, and perform genus classification.
#
# **Outputs:**
# - `outputs/phase_2/metadata/chm_assessment.json` (on Google Drive)
# - Publication-quality plots saved to `outputs/phase_2/figures/exp_02_chm`
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
from urban_tree_transfer.config import PROJECT_CRS, RANDOM_SEED
from urban_tree_transfer.utils import ExecutionLog, save_figure, setup_plotting
from urban_tree_transfer.utils.plotting import PUBLICATION_STYLE

from pathlib import Path
from datetime import datetime, timezone
import json

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, pearsonr, bootstrap

# Genus classification (deciduous/coniferous)
DECIDUOUS_GENERA = [
    "ACER", "AESCULUS", "AILANTHUS", "ALNUS", "BETULA", "CARPINUS",
    "CORNUS", "CORYLUS", "CRATAEGUS", "FAGUS", "FRAXINUS", "GLEDITSIA",
    "JUGLANS", "LIQUIDAMBAR", "MALUS", "PLATANUS", "POPULUS", "PRUNUS",
    "PYRUS", "QUERCUS", "ROBINIA", "SALIX", "SOPHORA", "SORBUS",
    "TILIA", "ULMUS",
]
CONIFEROUS_GENERA = [
    "ABIES", "LARIX", "PICEA", "PINUS", "TAXUS", "THUJA",
]

setup_plotting()
log = ExecutionLog("exp_02_chm_assessment")

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
FIGURES_DIR = OUTPUT_DIR / "figures" / "exp_02_chm"

CITIES = ["berlin", "leipzig"]
MIN_SAMPLES_PER_GENUS = 500
DETECTION_THRESHOLD_M = 2.0

for d in [METADATA_DIR, LOGS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Input:  {INPUT_DIR}")
print(f"Output: {METADATA_DIR}")
print(f"Plots:  {FIGURES_DIR}")
print(f"Cities: {CITIES}")
print(f"Random seed: {RANDOM_SEED}")

# %% [markdown]
# ## Data Loading & Validation
#
# Load Phase 2a feature data for Berlin and Leipzig and validate expected schema and CRS.
#

# %%
log.start_step("Data Loading & Validation")

try:
    city_data = {}
    required_cols = ["tree_id", "city", "genus_latin", "CHM_1m", "height_m", "plant_year", "geometry"]

    for city in CITIES:
        path = INPUT_DIR / f"trees_with_features_{city}.gpkg"
        if not path.exists():
            fallback_path = INPUT_DIR / f"trees_with_features_{city.title()}.gpkg"
            if fallback_path.exists():
                print(
                    "Warning: non-normalized city filename detected. "
                    f"Using {fallback_path.name} instead of {path.name}."
                )
                path = fallback_path
        print(f"Loading: {path}")
        df = gpd.read_file(path)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {city}: {missing}")

        if df.crs is None or df.crs.to_epsg() != int(str(PROJECT_CRS).split(":")[-1]):
            raise ValueError(f"Invalid CRS for {city}: {df.crs}. Expected {PROJECT_CRS}.")

        df["genus_latin"] = df["genus_latin"].astype(str).str.upper().str.strip()
        df["city"] = city

        city_data[city] = df
        print(f"Loaded {city}: {len(df):,} rows")

    log.end_step(status="success", records=sum(len(df) for df in city_data.values()))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Discriminative Power (ANOVA η²)
#
# Compute one-way ANOVA and η² for CHM_1m by genus per city.
#

# %%
log.start_step("Discriminative Power (ANOVA eta-squared)")

try:
    eta2_by_city = {}
    viable_genera_by_city = {}

    for city, df in city_data.items():
        df_valid = df[df["CHM_1m"].notna()].copy()
        genus_counts = df_valid["genus_latin"].value_counts()
        viable_genera = genus_counts[genus_counts >= MIN_SAMPLES_PER_GENUS].index.tolist()
        viable_genera_by_city[city] = sorted(viable_genera)

        groups = [
            df_valid[df_valid["genus_latin"] == g]["CHM_1m"].values
            for g in viable_genera
        ]
        if len(groups) < 2:
            raise ValueError(f"Not enough viable genera for ANOVA in {city}.")

        f_stat, p_value = f_oneway(*groups)
        grand_mean = df_valid["CHM_1m"].mean()
        ss_between = sum(
            len(df_valid[df_valid["genus_latin"] == g])
            * (df_valid[df_valid["genus_latin"] == g]["CHM_1m"].mean() - grand_mean) ** 2
            for g in viable_genera
        )
        ss_total = ((df_valid["CHM_1m"] - grand_mean) ** 2).sum()
        eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0

        eta2_by_city[city] = {
            "eta_squared": eta_squared,
            "f_stat": float(f_stat),
            "p_value": float(p_value),
            "n_genera": len(viable_genera),
            "n_samples": int(len(df_valid)),
        }

        print(f"{city.title()}: eta^2={eta_squared:.4f}, n_genera={len(viable_genera)}")

    log.end_step(status="success", records=len(eta2_by_city))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Cross-City Consistency (Cohen's d)
#
# Compute Cohen's d effect sizes and confidence intervals to assess cross-city CHM consistency.

# %%
def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d using pooled standard deviation."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.std(ddof=1) ** 2 + (n2 - 1) * group2.std(ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def cohens_d_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute Cohen's d with bootstrap CI."""
    def stat(g1, g2):
        # Return as 1D array instead of scalar to avoid concatenation error
        return np.atleast_1d(cohens_d(g1, g2))

    rng = np.random.default_rng(RANDOM_SEED)
    res = bootstrap(
        (group1, group2),
        stat,
        n_resamples=n_bootstrap,
        confidence_level=confidence,
        random_state=rng,
        method="percentile",
    )
    # Use .item() to extract scalar from 0-d array and avoid deprecation warning
    return res.confidence_interval.low.item(), res.confidence_interval.high.item()


log.start_step("Cross-City Consistency (Cohen's d)")

try:
    berlin = city_data["berlin"]
    leipzig = city_data["leipzig"]

    viable_common = sorted(
        set(viable_genera_by_city.get("berlin", []))
        & set(viable_genera_by_city.get("leipzig", []))
    )

    cohens_records = []
    for genus in viable_common:
        g1 = berlin[berlin["genus_latin"] == genus]["CHM_1m"].dropna().values
        g2 = leipzig[leipzig["genus_latin"] == genus]["CHM_1m"].dropna().values
        if len(g1) < MIN_SAMPLES_PER_GENUS or len(g2) < MIN_SAMPLES_PER_GENUS:
            continue

        d_value = cohens_d(g1, g2)
        ci_low, ci_high = cohens_d_ci(g1, g2)

        cohens_records.append(
            {
                "genus": genus,
                "cohens_d": d_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_berlin": int(len(g1)),
                "n_leipzig": int(len(g2)),
            }
        )

    cohens_df = pd.DataFrame(cohens_records).sort_values("cohens_d")
    mean_abs_d = float(cohens_df["cohens_d"].abs().mean()) if not cohens_df.empty else 0.0

    if mean_abs_d < 0.2:
        transfer_interpretation = "low transfer risk"
    elif mean_abs_d < 0.5:
        transfer_interpretation = "medium transfer risk"
    else:
        transfer_interpretation = "high transfer risk"

    print(f"Common viable genera: {len(viable_common)}")
    print(f"Mean |d|: {mean_abs_d:.3f} ({transfer_interpretation})")

    log.end_step(status="success", records=len(cohens_df))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Feature Engineering Validation
#
# Validate CHM_1m against cadastre height, and compute Z-score/percentile transforms.
#

# %%
log.start_step("Feature Engineering Validation")

try:
    combined = pd.concat(city_data.values(), ignore_index=True)
    combined = combined[combined["CHM_1m"].notna()].copy()

    valid_corr = combined[combined["height_m"].notna()].copy()
    r_value, p_value = pearsonr(valid_corr["CHM_1m"], valid_corr["height_m"])

    combined["CHM_1m_zscore"] = combined.groupby("genus_latin")["CHM_1m"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
    )
    combined["CHM_1m_percentile"] = combined.groupby("genus_latin")["CHM_1m"].transform(
        lambda x: x.rank(pct=True) * 100.0
    )

    print(f"CHM vs height correlation: r={r_value:.3f} (p={p_value:.3g})")

    log.end_step(status="success", records=len(combined))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Plant Year Threshold Analysis
#
# Determine the latest planting year with median CHM above the detection threshold.
#

# %%
log.start_step("Plant Year Threshold Analysis")

try:
    df_with_year = combined[combined["plant_year"].notna()].copy()
    df_with_year["plant_year"] = df_with_year["plant_year"].astype(int)

    median_by_year = (
        df_with_year.groupby("plant_year")["CHM_1m"]
        .median()
        .sort_index()
    )

    min_year = int(median_by_year.index.min())
    max_year = int(median_by_year.index.max())
    years_below = median_by_year[median_by_year < DETECTION_THRESHOLD_M]
    if not years_below.empty:
        cutoff_year = int(years_below.index.min())
        recommended_max_year = max(min_year, cutoff_year - 1)
    else:
        recommended_max_year = max_year

    df_with_year["year_cohort"] = (df_with_year["plant_year"] // 2) * 2

    print(f"Recommended max plant year: {recommended_max_year}")

    log.end_step(status="success", records=len(median_by_year))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Genus Inventory & Classification
#
# Create genus inventory, classify deciduous/coniferous, and determine analysis scope.
#

# %%
log.start_step("Genus Inventory & Classification")

try:
    genus_inventory = {
        city: city_data[city]["genus_latin"].value_counts().to_dict()
        for city in CITIES
    }

    combined_counts = pd.Series(genus_inventory["berlin"]).add(
        pd.Series(genus_inventory["leipzig"]), fill_value=0
    ).sort_values(ascending=False)

    def classify_genus(genus: str) -> str:
        if genus in DECIDUOUS_GENERA:
            return "deciduous"
        if genus in CONIFEROUS_GENERA:
            return "coniferous"
        return "unknown"

    genus_types = {g: classify_genus(g) for g in combined_counts.index}
    unclassified_genera = [g for g, t in genus_types.items() if t == "unknown"]

    viable_genera_overall = [
        g for g, count in combined_counts.items() if count >= MIN_SAMPLES_PER_GENUS
    ]
    conifer_genera_count = len([g for g in viable_genera_overall if g in CONIFEROUS_GENERA])
    conifer_sample_count = int(
        combined[combined["genus_latin"].isin(CONIFEROUS_GENERA)]["genus_latin"].count()
    )

    if conifer_genera_count < 3 or conifer_sample_count < 500:
        analysis_scope = "deciduous_only"
        reason = (
            f"n_genera={conifer_genera_count} < 3"
            if conifer_genera_count < 3
            else f"n_samples={conifer_sample_count} < 500"
        )
        include_conifers = False
    else:
        analysis_scope = "all"
        reason = "Sufficient conifer diversity and samples"
        include_conifers = True

    print(f"Analysis scope: {analysis_scope} ({reason})")

    log.end_step(status="success", records=len(combined_counts))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Visualizations
#
# Generate required plots for CHM assessment.
#

# %%
log.start_step("Visualizations")

try:
    dpi = PUBLICATION_STYLE.get("dpi_export", 300)

    # Plot 1: CHM Distribution by Genus (faceted by city)
    plot_data = pd.concat(city_data.values(), ignore_index=True)
    plot_data = plot_data[plot_data["CHM_1m"].notna()].copy()
    common_genera = sorted(
        set(viable_genera_by_city.get("berlin", []))
        & set(viable_genera_by_city.get("leipzig", []))
    )
    plot_data = plot_data[plot_data["genus_latin"].isin(common_genera)]

    g = sns.catplot(
        data=plot_data,
        x="genus_latin",
        y="CHM_1m",
        col="city",
        kind="violin",
        height=6,
        aspect=1.4,
        order=common_genera,
        cut=0,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Genus", "CHM_1m (m)")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=45)
    save_figure(g.fig, FIGURES_DIR / "chm_boxplot_per_genus.png", dpi=dpi)

    # Plot 2: CHM vs Cadastre Correlation
    fig, ax = plt.subplots(figsize=PUBLICATION_STYLE["figsize"])
    city_colors = {"berlin": "#1f77b4", "leipzig": "#ff7f0e"}
    for city in CITIES:
        subset = city_data[city]
        subset = subset[subset["CHM_1m"].notna() & subset["height_m"].notna()]
        sns.regplot(
            data=subset,
            x="height_m",
            y="CHM_1m",
            scatter_kws={"alpha": 0.3},
            line_kws={"linewidth": 2},
            color=city_colors.get(city, None),
            ax=ax,
            label=city.title(),
        )
    ax.set_title("CHM vs Cadastre Height", fontsize=14)
    ax.set_xlabel("Cadastre height (m)")
    ax.set_ylabel("CHM_1m (m)")
    ax.legend(loc="best")
    ax.text(0.02, 0.95, f"r = {r_value:.2f}", transform=ax.transAxes)
    save_figure(fig, FIGURES_DIR / "chm_cadastre_correlation.png", dpi=dpi)

    # Plot 3: Discriminative Power (eta^2) Comparison
    fig, ax = plt.subplots(figsize=PUBLICATION_STYLE["figsize"])
    eta_values = [eta2_by_city[c]["eta_squared"] for c in CITIES]
    ax.bar([c.title() for c in CITIES], eta_values, color=["#1f77b4", "#ff7f0e"])
    ax.axhline(0.06, color="gray", linestyle="--", linewidth=1, label="Medium (0.06)")
    ax.axhline(0.14, color="black", linestyle=":", linewidth=1, label="Large (0.14)")
    ax.set_ylabel("Eta-squared (η²)")
    ax.set_title("Discriminative Power by City")
    ax.legend(loc="upper right")
    save_figure(fig, FIGURES_DIR / "eta2_comparison.png", dpi=dpi)

    # Plot 4: Cohen's d Forest Plot
    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * len(cohens_df))))
    if not cohens_df.empty:
        y_pos = np.arange(len(cohens_df))
        ax.errorbar(
            cohens_df["cohens_d"],
            y_pos,
            xerr=[
                cohens_df["cohens_d"] - cohens_df["ci_low"],
                cohens_df["ci_high"] - cohens_df["cohens_d"],
            ],
            fmt="o",
            color="#1f77b4",
            ecolor="gray",
            capsize=3,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cohens_df["genus"])
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(0.2, color="gray", linestyle="--", linewidth=1)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.axvline(0.8, color="gray", linestyle="-.", linewidth=1)
        ax.set_xlabel("Cohen's d (Berlin - Leipzig)")
        ax.set_title("Cohen's d by Genus (with 95% CI)")
    save_figure(fig, FIGURES_DIR / "cohens_d_forest_plot.png", dpi=dpi)

    # Plot 5: CHM Distribution Comparison (Cities)
    fig, ax = plt.subplots(figsize=PUBLICATION_STYLE["figsize"])
    for city in CITIES:
        subset = city_data[city]
        subset = subset[subset["CHM_1m"].notna()]
        sns.histplot(
            subset["CHM_1m"],
            bins=30,
            kde=True,
            stat="density",
            alpha=0.35,
            label=city.title(),
            ax=ax,
        )
    ax.set_title("CHM Distribution: Berlin vs Leipzig")
    ax.set_xlabel("CHM_1m (m)")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    save_figure(fig, FIGURES_DIR / "chm_distribution_cities.png", dpi=dpi)

    # Plot 6: CHM vs Plant Year (improved readability)
    fig, ax = plt.subplots(figsize=(12, 6))
    if not df_with_year.empty:
        # Filter to years with sufficient data (>50 samples) and last 50 years
        year_counts = df_with_year.groupby("plant_year").size()
        valid_years = year_counts[year_counts >= 50].index
        max_available_year = int(df_with_year["plant_year"].max())
        cutoff_year = max(max_available_year - 50, valid_years.min() if len(valid_years) > 0 else max_available_year - 50)
        
        plot_years = sorted([y for y in valid_years if y >= cutoff_year])
        
        # Compute median and quartiles by year and city
        city_colors_map = {"berlin": "#1f77b4", "leipzig": "#ff7f0e"}
        
        for city in CITIES:
            city_df = df_with_year[df_with_year["city"] == city]
            city_df = city_df[city_df["plant_year"].isin(plot_years)]
            
            if not city_df.empty:
                stats_by_year = city_df.groupby("plant_year")["CHM_1m"].agg([
                    ("median", "median"),
                    ("q25", lambda x: x.quantile(0.25)),
                    ("q75", lambda x: x.quantile(0.75)),
                ]).reset_index()
                
                ax.plot(
                    stats_by_year["plant_year"],
                    stats_by_year["median"],
                    marker="o",
                    linewidth=2,
                    label=f"{city.title()}",
                    color=city_colors_map[city],
                )
                ax.fill_between(
                    stats_by_year["plant_year"],
                    stats_by_year["q25"],
                    stats_by_year["q75"],
                    alpha=0.2,
                    color=city_colors_map[city],
                )
        
        ax.axhline(
            DETECTION_THRESHOLD_M,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Detection threshold ({DETECTION_THRESHOLD_M}m)",
        )
        ax.axvline(
            recommended_max_year,
            color="darkred",
            linestyle=":",
            linewidth=2,
            label=f"Recommended cutoff ({recommended_max_year})",
        )
        
        ax.set_title("CHM Height by Plant Year (Median with IQR)")
        ax.set_xlabel("Plant year")
        ax.set_ylabel("CHM_1m (m)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        # Show every 5th year on x-axis for readability
        if len(plot_years) > 15:
            tick_years = [y for y in plot_years if y % 5 == 0]
            ax.set_xticks(tick_years)
            ax.set_xticklabels(tick_years, rotation=45)
    
    save_figure(fig, FIGURES_DIR / "chm_vs_plant_year.png", dpi=dpi)

    # Plot 7: Genus Inventory (Top 20 genera for readability)
    fig, ax = plt.subplots(figsize=(14, 7))
    top_n = 20
    top_genera = combined_counts.head(top_n)
    
    # Define colors for genus types (not cities)
    genus_type_colors = {
        "deciduous": "#2ecc71",  # Green for deciduous
        "coniferous": "#3498db",  # Blue for coniferous
        "unknown": "#95a5a6"      # Gray for unknown
    }
    colors_list = [genus_type_colors.get(genus_types[g], "#7f7f7f") for g in top_genera.index]

    ax.bar(range(len(top_genera)), top_genera.values, color=colors_list)
    ax.set_xticks(range(len(top_genera)))
    ax.set_xticklabels(top_genera.index, rotation=45, ha="right")
    ax.axhline(
        MIN_SAMPLES_PER_GENUS,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Min samples ({MIN_SAMPLES_PER_GENUS})",
    )
    ax.set_title(f"Genus Inventory (Top {top_n} by Sample Count)")
    ax.set_xlabel("Genus")
    ax.set_ylabel("Sample count")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=genus_type_colors["deciduous"], label="Deciduous"),
        Patch(facecolor=genus_type_colors["coniferous"], label="Coniferous"),
    ]
    if any(t == "unknown" for t in genus_types.values()):
        legend_elements.append(Patch(facecolor=genus_type_colors["unknown"], label="Unclassified"))
    ax.legend(handles=legend_elements, loc="upper right")

    save_figure(fig, FIGURES_DIR / "genus_inventory.png", dpi=dpi)


    log.end_step(status="success", records=7)

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
log.start_step("Output Validation")

try:
    # Validate eta^2 values
    for city, stats in eta2_by_city.items():
        eta = stats["eta_squared"]
        if not (0.0 <= eta <= 1.0):
            raise ValueError(f"Invalid eta^2 for {city}: {eta} (must be in [0, 1])")

    # Validate Cohen's d CIs
    if not cohens_df.empty:
        invalid_ci = cohens_df[cohens_df["ci_low"] > cohens_df["ci_high"]]
        if not invalid_ci.empty:
            raise ValueError(
                f"Invalid CI bounds for genera: {invalid_ci['genus'].tolist()}"
            )

    # Validate correlation
    if not (-1.0 <= r_value <= 1.0):
        raise ValueError(f"Invalid correlation: {r_value} (must be in [-1, 1])")

    # Validate plant year threshold
    if not (min_year <= recommended_max_year <= max_year):
        raise ValueError(
            f"Plant year threshold {recommended_max_year} outside data range [{min_year}, {max_year}]"
        )

    # Validate all genera classified
    if unclassified_genera:
        print(
            f"Warning: {len(unclassified_genera)} unclassified genera: {unclassified_genera[:5]}"
        )

    print("✓ All validation checks passed")
    log.end_step(status="success", records=4)

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## JSON Output
#
# Save CHM assessment configuration for downstream use.
#

# %%
log.start_step("JSON Output")

try:
    include_chm = (
        eta2_by_city.get("berlin", {}).get("eta_squared", 0.0) >= 0.06
        and eta2_by_city.get("leipzig", {}).get("eta_squared", 0.0) >= 0.06
        and mean_abs_d < 0.2
    )

    median_chm_by_year = {str(int(k)): float(v) for k, v in median_by_year.items()}
    justification = (
        f"Trees planted after {recommended_max_year} have median CHM < "
        f"{DETECTION_THRESHOLD_M}m in 2021 imagery"
    )

    output = {
        "analysis_date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "chm_features": ["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"],
        "include_chm": include_chm,
        "discriminative_power": {
            "berlin_eta2": round(eta2_by_city["berlin"]["eta_squared"], 4),
            "leipzig_eta2": round(eta2_by_city["leipzig"]["eta_squared"], 4),
        },
        "transfer_risk": {
            "cohens_d_mean": round(mean_abs_d, 4),
            "interpretation": transfer_interpretation,
        },
        "validation": {
            "chm_cadastre_correlation": round(float(r_value), 4),
        },
        "plant_year_analysis": {
            "detection_threshold_m": DETECTION_THRESHOLD_M,
            "median_chm_by_year": median_chm_by_year,
            "recommended_max_plant_year": int(recommended_max_year),
            "justification": justification,
        },
        "genus_inventory": {
            "berlin": {k: int(v) for k, v in genus_inventory["berlin"].items()},
            "leipzig": {k: int(v) for k, v in genus_inventory["leipzig"].items()},
            "classification": {
                "deciduous": DECIDUOUS_GENERA,
                "coniferous": CONIFEROUS_GENERA,
            },
            "unclassified_genera": unclassified_genera,
            "conifer_analysis": {
                "n_genera": int(conifer_genera_count),
                "n_samples": int(conifer_sample_count),
                "include_in_analysis": bool(include_conifers),
                "reason": reason,
            },
            "analysis_scope": analysis_scope,
        },
    }

    output_path = METADATA_DIR / "chm_assessment.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")

    log.end_step(status="success", records=1)

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
print("=" * 60)
print("CHM ASSESSMENT SUMMARY")
print("=" * 60)
print(f"\n1. DISCRIMINATIVE POWER")
for city in CITIES:
    eta = eta2_by_city[city]["eta_squared"]
    interp = "large" if eta >= 0.14 else "medium" if eta >= 0.06 else "small"
    print(f"   {city.title():8s}: η² = {eta:.4f} ({interp} effect)")

print(f"\n2. TRANSFER RISK")
print(f"   Mean |Cohen's d| = {mean_abs_d:.3f}")
print(f"   Interpretation: {transfer_interpretation}")
print(f"   Common viable genera: {len(viable_common)}")

print(f"\n3. VALIDATION")
print(f"   CHM-Cadastre correlation: r = {r_value:.3f}")
interp_corr = "moderate (good)" if 0.4 <= abs(r_value) <= 0.6 else "low/high (check)"
print(f"   Interpretation: {interp_corr}")

print(f"\n4. PLANT YEAR THRESHOLD")
print(f"   Detection threshold: {DETECTION_THRESHOLD_M}m")
print(f"   Recommended max plant year: {recommended_max_year}")
print(f"   Years analyzed: {min_year} - {max_year}")

print(f"\n5. GENUS CLASSIFICATION")
print(f"   Analysis scope: {analysis_scope}")
print(f"   Conifer genera (viable): {conifer_genera_count}")
print(f"   Conifer samples (total): {conifer_sample_count}")
print(f"   Include CHM features: {include_chm}")

print(f"\n6. OUTPUT FILES")
print(f"   JSON: chm_assessment.json")
print(f"   Plots: {len(list(FIGURES_DIR.glob('*.png')))} PNG files")

print("\nNote: Legacy pipeline had cross-tree contamination (r=0.638).")
print("      Current pipeline: r={:.3f} (no contamination).".format(r_value))
print("=" * 60)


# %% [markdown]
# ## Summary & Manual Sync Instructions
#

# %%
# Save execution log
log.summary()
log_path = LOGS_DIR / f"{log.notebook}_execution.json"
log.save(log_path)
print(f"Execution log saved: {log_path}")

print("\n--- JSON OUTPUT ---")
print(f"  {output_path.name}")

print("\n--- PLOTS CREATED ---")
for f in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  {f.name}")

print("\n" + "="*70)
print("⚠️  MANUAL SYNC REQUIRED:")
print("="*70)
print(f"1. JSON: outputs/phase_2/metadata/chm_assessment.json")
print(f"2. Plots: outputs/phase_2/figures/exp_02_chm/*.png")
print(f"3. Copy from Drive to local repo")
print(f"4. git add outputs/phase_2/metadata/chm_assessment.json")
print(f"5. git add outputs/phase_2/figures/exp_02_chm/")
print(f"6. git commit -m 'feat(exploratory): add CHM assessment analysis'")
print("="*70)
