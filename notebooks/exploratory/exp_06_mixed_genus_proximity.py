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
from urban_tree_transfer.feature_engineering.proximity import apply_proximity_filter

from pathlib import Path
from datetime import datetime, timezone
import json

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle

setup_plotting()
log = ExecutionLog("exp_06_proximity")

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
FIGURES_DIR = OUTPUT_DIR / "figures" / "exp_06_proximity"

CITIES = ["berlin", "leipzig"]
THRESHOLDS_TO_TEST = [5, 10, 15, 20, 30]  # meters

for d in [METADATA_DIR, LOGS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Input (Phase 2):  {INPUT_DIR}")
print(f"Output (JSONs):   {METADATA_DIR}")
print(f"Figures:          {FIGURES_DIR}")
print(f"Logs (Drive):     {LOGS_DIR}")
print(f"Cities:           {CITIES}")
print(f"Random seed:      {RANDOM_SEED}")



# %%
# ============================================================
# SECTION 1: Data Loading & Validation
# ============================================================

log.start_step("Data Loading")

required_columns = {"geometry", "genus_latin", "city"}
city_data = {}

total_trees = 0

for city in CITIES:
    path = INPUT_DIR / f"trees_clean_{city}.gpkg"
    print(f"Loading: {path}")
    gdf = gpd.read_file(path)

    # Ensure city column exists
    if "city" not in gdf.columns:
        gdf["city"] = city
        print(f"  Added missing city column for {city}.")

    missing = required_columns - set(gdf.columns)
    if missing:
        raise ValueError(f"Missing required columns for {city}: {missing}")

    # Validate CRS (EPSG:25833)
    if gdf.crs is None:
        raise ValueError(f"CRS missing for {city}. Expected {PROJECT_CRS}.")

    if gdf.crs.to_string() != PROJECT_CRS:
        print(f"  Reprojecting {city} to {PROJECT_CRS} (was {gdf.crs.to_string()})")
        gdf = gdf.to_crs(PROJECT_CRS)

    gdf = gdf[gdf["genus_latin"].notna()].copy()

    print(f"  {city}: {len(gdf):,} rows")
    city_data[city] = gdf
    total_trees += len(gdf)

log.end_step(status="success", records=total_trees)


# %% [markdown]
# ## Section 2: Geometric Definition and Pixel Footprint Analysis
#
# Klarstellung der geometrischen Interpretation des Proximity-Thresholds im Kontext
# der Sentinel-2 Pixelauflosung.
#
# **Sentinel-2 Kontext:**
# - Pixelgrose: 10m x 10m (B2, B3, B4, B8)
# - Pixel-Diagonale: ~14.1m
# - Baumposition: Punktgeometrie (Zentroid)
#
# **Distanz-Metrik:**
# - Euklidische Distanz zwischen Baum-Zentroiden (Punkt-zu-Punkt)
# - 20m Threshold ≈ 2-Pixel Separation (Edge-to-Edge Kontakt an der Grenze)
#
# **Kontaminationszonen:**
# - < 10m: Vollstandige Pixel-Uberlappung (HIGH)
# - 10-15m: Partielle Uberlappung (MEDIUM)
# - 15-20m: Edge-Contact (LOW)
# - > 20m: Keine Uberlappung (NONE)
#

# %%
# ============================================================
# SECTION 3: Nearest Different-Genus Distance (OPTIMIZED)
# ============================================================
from scipy.spatial import cKDTree

log.start_step("Nearest Neighbor Analysis")

np.random.seed(RANDOM_SEED)

def compute_nearest_diff_genus_fast(gdf: gpd.GeoDataFrame, genus_col: str = "genus_latin", k: int = 50) -> pd.Series:
    """
    Computes distance to the nearest neighbor of a different genus using cKDTree.
    Optimization: O(N log N) instead of O(N^2) using Spatial Indexing.
    """
    # 1. Prepare coordinates and genus codes
    # We extract X/Y coordinates for the cKDTree
    coords = np.column_stack((gdf.geometry.x, gdf.geometry.y))
    
    # Convert genus strings to integer codes for faster comparison
    genus_codes, _ = pd.factorize(gdf[genus_col])
    
    # 2. Build spatial index (extremely fast for point data)
    print(f"  Building spatial index for {len(gdf):,} trees...")
    tree = cKDTree(coords)
    
    # 3. Query k nearest neighbors (k+1 to include the point itself)
    # asking for 50 neighbors is usually enough to find a different genus in urban settings
    print("  Querying nearest neighbors...")
    dists, idxs = tree.query(coords, k=k+1, workers=-1)
    
    # 4. Find first neighbor with different genus
    # Retrieve the genus codes of the neighbors found
    neighbor_genera = genus_codes[idxs]
    self_genera = genus_codes.reshape(-1, 1)
    
    # Boolean matrix: True where neighbor genus is DIFFERENT from self
    is_diff = neighbor_genera != self_genera
    
    # Find the index of the first True value in each row
    # argmax returns the first index of the max value (True=1). 
    # If no True is found (all neighbors are same genus), it returns 0.
    first_diff_idx = np.argmax(is_diff, axis=1)
    
    # 5. Extract distances
    # Initialize with infinity (for cases where no diff genus is found within k neighbors)
    nearest_dists = np.full(len(gdf), np.inf)
    
    # valid_mask: where a different genus was actually found
    # (index > 0 because index 0 is the tree itself/same genus)
    valid_mask = first_diff_idx > 0
    
    # Assign distances using advanced indexing
    nearest_dists[valid_mask] = dists[valid_mask, first_diff_idx[valid_mask]]
    
    return pd.Series(nearest_dists, index=gdf.index)

for city, gdf in city_data.items():
    print(f"Computing nearest different-genus distance: {city} (Optimized)")
    city_data[city] = gdf.copy()
    city_data[city]["nearest_diff_genus_m"] = compute_nearest_diff_genus_fast(city_data[city])

log.end_step(status="success")

# %%
# ============================================================
# SECTION 4: Distance Distribution Analysis
# ============================================================

log.start_step("Distance Distribution")

stats_by_city = {}

for city, gdf in city_data.items():
    distances = gdf["nearest_diff_genus_m"].to_numpy()
    distances = distances[np.isfinite(distances)]

    if len(distances) == 0:
        raise ValueError(f"No finite distances for {city}. Check input data.")

    percentiles = np.percentile(distances, [10, 25, 50, 75, 90, 95, 99])
    stats_by_city[city] = {
        "count": int(len(distances)),
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
    }

# Visualization 1: Histogram per city
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
city_colors = {"berlin": "#1f77b4", "leipzig": "#ff7f0e"}

for idx, city in enumerate(CITIES):
    ax = axes[idx]
    distances = city_data[city]["nearest_diff_genus_m"]
    distances = distances[np.isfinite(distances)]

    sns.histplot(distances, bins=40, color=city_colors[city], ax=ax)
    for t in THRESHOLDS_TO_TEST:
        ax.axvline(t, color="black", linestyle="--", linewidth=1)

    ax.set_title(f"{city.title()}")
    ax.set_xlabel("Nearest different-genus distance (m)")
    ax.set_ylabel("Count")

fig.suptitle("Distribution of Nearest Different-Genus Distance", fontsize=14, fontweight="bold")
plt.tight_layout()
save_figure(fig, FIGURES_DIR / "distance_distribution.png")

log.end_step(status="success")


# %%
# ============================================================
# SECTION 5: Threshold Sensitivity Analysis
# ============================================================

log.start_step("Threshold Sensitivity")

rows = []
for city, gdf in city_data.items():
    total = len(gdf)
    for threshold in THRESHOLDS_TO_TEST:
        removed = int((gdf["nearest_diff_genus_m"] < threshold).sum())
        kept = total - removed
        retention = kept / total if total else 0.0
        rows.append(
            {
                "city": city,
                "threshold": threshold,
                "trees_removed": removed,
                "trees_kept": kept,
                "retention_rate": retention,
            }
        )

sensitivity_results = pd.DataFrame(rows)

# Visualization 2: Retention Rate per Threshold
fig, ax = plt.subplots(figsize=PUBLICATION_STYLE["figsize"])

for city in CITIES:
    data = sensitivity_results[sensitivity_results["city"] == city]
    ax.plot(
        data["threshold"],
        data["retention_rate"],
        marker="o",
        label=city.title(),
    )

ax.axhline(0.85, color="red", linestyle="--", label="85% retention target")
ax.set_title("Retention Rate vs Proximity Threshold")
ax.set_xlabel("Threshold (m)")
ax.set_ylabel("Retention Rate")
ax.set_ylim(0, 1.0)
ax.legend(loc="best")

save_figure(fig, FIGURES_DIR / "retention_rate_sensitivity.png")

log.end_step(status="success")


# %%
# ============================================================
# SECTION 5B: Threshold Sensitivity Curve
# ============================================================

log.start_step("Threshold Sensitivity Curve")

try:
    thresholds = [5, 10, 15, 20, 25, 30]
    sensitivity_rows = []

    for threshold in thresholds:
        filtered_gdf, stats = apply_proximity_filter(
            city_data["berlin"].copy(),
            threshold_m=threshold,
        )
        sensitivity_rows.append({
            "threshold_m": threshold,
            "retention_rate": stats["retention_rate"],
            "removed_count": stats["removed_count"],
            "retained_count": stats["retained_count"],
        })

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        sensitivity_df["threshold_m"],
        sensitivity_df["retention_rate"],
        "o-",
        linewidth=2,
        markersize=8,
        color="steelblue",
    )

    ax.axvline(20, color="red", linestyle="--", linewidth=2, label="Recommended: 20m")
    retention_at_20 = sensitivity_df[sensitivity_df["threshold_m"] == 20]["retention_rate"].values[0]
    ax.axhline(retention_at_20, color="red", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("Proximity Threshold (m)", fontsize=12)
    ax.set_ylabel("Retention Rate", fontsize=12)
    ax.set_title("Threshold Sensitivity: Retention Rate vs. Proximity Threshold", fontsize=14)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    ax.annotate(
        f"{retention_at_20:.1%} retained",
        xy=(20, retention_at_20),
        xytext=(22, retention_at_20 - 0.05),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red",
    )

    plt.tight_layout()
    save_figure(fig, FIGURES_DIR / "threshold_sensitivity_curve.png", dpi=300)
    print(f"Saved: {FIGURES_DIR / 'threshold_sensitivity_curve.png'}")

    log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 6: Genus-Specific Impact
# ============================================================

log.start_step("Genus-Specific Impact")

pixel_size_m = 10
min_pixel_coverage = 2
min_threshold_for_pixels = pixel_size_m * min_pixel_coverage

threshold_evaluations = []

for threshold in THRESHOLDS_TO_TEST:
    evaluation = {"threshold": threshold, "per_city": {}}
    for city, gdf in city_data.items():
        gdf = gdf.copy()
        gdf["removed"] = gdf["nearest_diff_genus_m"] < threshold

        genus_stats = (
            gdf.groupby("genus_latin")["removed"]
            .agg(total_trees="count", removed="sum")
            .reset_index()
        )
        genus_stats["removal_rate"] = genus_stats["removed"] / genus_stats["total_trees"]

        max_deviation = float(genus_stats["removal_rate"].max() - genus_stats["removal_rate"].min())
        retention_rate = float(
            sensitivity_results[
                (sensitivity_results["city"] == city)
                & (sensitivity_results["threshold"] == threshold)
            ]["retention_rate"].iloc[0]
        )

        evaluation["per_city"][city] = {
            "retention_rate": retention_rate,
            "max_deviation": max_deviation,
            "genus_stats": genus_stats,
        }

    evaluation["passes_retention"] = all(
        evaluation["per_city"][city]["retention_rate"] >= 0.85 for city in CITIES
    )
    evaluation["passes_uniformity"] = all(
        evaluation["per_city"][city]["max_deviation"] < 0.10 for city in CITIES
    )
    evaluation["covers_two_pixels"] = threshold >= min_threshold_for_pixels

    threshold_evaluations.append(evaluation)

# Select best passing threshold: smallest threshold that meets all criteria
passing_thresholds = [
    e for e in threshold_evaluations
    if e["passes_retention"] and e["passes_uniformity"] and e["covers_two_pixels"]
]

if passing_thresholds:
    recommended = sorted(passing_thresholds, key=lambda x: x["threshold"])[0]
    recommended_threshold = recommended["threshold"]
else:
    # Fallback: choose threshold with highest average retention
    avg_retention = []
    for e in threshold_evaluations:
        avg = np.mean([e["per_city"][c]["retention_rate"] for c in CITIES])
        avg_retention.append((avg, e["threshold"]))
    recommended_threshold = max(avg_retention)[1]

print(f"Recommended threshold: {recommended_threshold}m")

# Compute genus impact for recommended threshold
impact_rows = []
for city, gdf in city_data.items():
    gdf = gdf.copy()
    gdf["removed"] = gdf["nearest_diff_genus_m"] < recommended_threshold
    genus_stats = (
        gdf.groupby("genus_latin")["removed"]
        .agg(total_trees="count", removed="sum")
        .reset_index()
    )
    genus_stats["removal_rate"] = genus_stats["removed"] / genus_stats["total_trees"]
    genus_stats["city"] = city
    impact_rows.append(genus_stats)

genus_impact = pd.concat(impact_rows, ignore_index=True)

# Visualization 3: Removal Rate per Genus (Grouped Bar Chart)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for idx, city in enumerate(CITIES):
    ax = axes[idx]
    subset = genus_impact[genus_impact["city"] == city].sort_values("removal_rate")
    ax.bar(subset["genus_latin"], subset["removal_rate"], color=city_colors[city])
    ax.axhline(subset["removal_rate"].mean(), color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{city.title()}")
    ax.set_xlabel("Genus")
    ax.set_ylabel("Removal rate")
    ax.tick_params(axis="x", rotation=90)

fig.suptitle(
    f"Genus-Specific Removal Rate ({recommended_threshold}m threshold)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
save_figure(fig, FIGURES_DIR / "genus_specific_impact.png")

max_deviation = genus_impact.groupby("city")["removal_rate"].apply(lambda x: x.max() - x.min())
is_uniform = max_deviation < 0.10

log.end_step(status="success")


# %%
# ============================================================
# SECTION 7: Spatial Distribution
# ============================================================

log.start_step("Spatial Distribution")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, city in enumerate(CITIES):
    ax = axes[idx]
    gdf = city_data[city].copy()
    gdf["affected"] = gdf["nearest_diff_genus_m"] < recommended_threshold

    gdf.plot(ax=ax, color="lightgray", markersize=2)
    gdf[gdf["affected"]].plot(ax=ax, color="red", markersize=6, label="Affected")

    ax.set_title(f"{city.title()}")
    ax.set_axis_off()

fig.suptitle(
    f"Spatial Distribution of Affected Trees ({recommended_threshold}m threshold)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
fig_name = f"spatial_distribution_{recommended_threshold}m.png"
save_figure(fig, FIGURES_DIR / fig_name)

log.end_step(status="success")


# %%
# ============================================================
# SECTION 7B: Proximity Zone Contamination Map (PRD 002d Improvement 5)
# ============================================================

log.start_step("Proximity Zone Contamination Map")

# Define contamination zones based on pixel overlap
def assign_contamination_zone(distance):
    """Assign contamination zone based on nearest different-genus distance."""
    if distance < 10:
        return "high"
    elif distance < 15:
        return "medium"
    elif distance < 20:
        return "low"
    else:
        return "none"

# Color scheme for contamination zones
zone_colors = {
    "high": "#d62728",      # Red - full pixel overlap
    "medium": "#ff7f0e",    # Orange - partial overlap
    "low": "#ffbb78",       # Light orange - edge contact
    "none": "#2ca02c"       # Green - no contamination
}

zone_labels = {
    "high": "HIGH (<10m, full overlap)",
    "medium": "MEDIUM (10-15m, partial overlap)",
    "low": "LOW (15-20m, edge contact)",
    "none": "NONE (>20m, clean)"
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, city in enumerate(CITIES):
    ax = axes[idx]
    gdf = city_data[city].copy()

    # Assign contamination zones
    gdf["contamination_zone"] = gdf["nearest_diff_genus_m"].apply(assign_contamination_zone)

    # Plot each zone separately for proper legend
    for zone in ["high", "medium", "low", "none"]:
        zone_trees = gdf[gdf["contamination_zone"] == zone]
        if len(zone_trees) > 0:
            zone_trees.plot(
                ax=ax,
                color=zone_colors[zone],
                markersize=3 if zone == "none" else 8,
                alpha=0.6 if zone == "none" else 0.8,
                label=zone_labels[zone]
            )

    # Calculate zone statistics
    zone_counts = gdf["contamination_zone"].value_counts()
    total = len(gdf)

    stats_text = "Contamination Zones:\n"
    for zone in ["high", "medium", "low", "none"]:
        count = zone_counts.get(zone, 0)
        pct = (count / total * 100) if total > 0 else 0
        stats_text += f"{zone.upper()}: {count} ({pct:.1f}%)\n"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_title(f"{city.title()}", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)

fig.suptitle(
    "Spatial Contamination Map: Sentinel-2 Pixel Overlap Zones\n"
    "(Point-to-Point Distance Between Different Genera)",
    fontsize=13,
    fontweight="bold"
)
plt.tight_layout()

save_figure(fig, FIGURES_DIR / "spatial_contamination_map.png", dpi=300)

log.end_step(status="success")

# %%
# ============================================================
# SECTION 8: Pixel Footprint Visualization
# ============================================================

import matplotlib.patches as patches

log.start_step("Pixel Footprint Visualization")

try:
    def plot_pixel_scenario(distance_m, ax):
        """Plot two trees at given distance with Sentinel-2 pixel footprints."""
        tree_a = (0, 0)
        tree_b = (distance_m, 0)

        ax.plot(tree_a[0], tree_a[1], "ro", markersize=12, label="Tree A (QUERCUS)", zorder=3)
        ax.plot(tree_b[0], tree_b[1], "go", markersize=12, label="Tree B (TILIA)", zorder=3)

        pixel_a = patches.Rectangle(
            (tree_a[0] - 5, tree_a[1] - 5),
            10,
            10,
            linewidth=2,
            edgecolor="red",
            facecolor="red",
            alpha=0.2,
            label="Pixel A",
        )
        pixel_b = patches.Rectangle(
            (tree_b[0] - 5, tree_b[1] - 5),
            10,
            10,
            linewidth=2,
            edgecolor="green",
            facecolor="green",
            alpha=0.2,
            label="Pixel B",
        )
        ax.add_patch(pixel_a)
        ax.add_patch(pixel_b)

        overlap_start = max(tree_a[0] - 5, tree_b[0] - 5)
        overlap_end = min(tree_a[0] + 5, tree_b[0] + 5)

        if overlap_end > overlap_start:
            overlap_width = overlap_end - overlap_start
            overlap = patches.Rectangle(
                (overlap_start, -5),
                overlap_width,
                10,
                linewidth=0,
                facecolor="yellow",
                alpha=0.6,
                zorder=2,
            )
            ax.add_patch(overlap)
            ax.text(
                (overlap_start + overlap_end) / 2,
                6.5,
                "Spectral\nContamination",
                ha="center",
                fontsize=9,
                color="darkorange",
                fontweight="bold",
            )

        ax.plot([tree_a[0], tree_b[0]], [tree_a[1], tree_b[1]], "k--", linewidth=1, zorder=1)
        ax.text(
            distance_m / 2,
            -1.5,
            f"{distance_m}m",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xlim(-10, distance_m + 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        ax.set_xlabel("Distance (m)", fontsize=10)
        ax.set_ylabel("Distance (m)", fontsize=10)
        ax.grid(True, alpha=0.3)

        if distance_m < 10:
            status = "HIGH Contamination"
            color = "red"
        elif distance_m < 15:
            status = "MEDIUM Contamination"
            color = "orange"
        elif distance_m < 20:
            status = "LOW Contamination"
            color = "yellow"
        else:
            status = "NO Contamination"
            color = "green"

        ax.set_title(f"{distance_m}m Distance\n{status}", fontsize=11, color=color, fontweight="bold")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    distances = [10, 15, 20, 25]
    for idx, dist in enumerate(distances):
        plot_pixel_scenario(dist, axes[idx])

    plt.suptitle(
        "Sentinel-2 Pixel Contamination Scenarios\n(10m x 10m pixel footprints)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    save_figure(fig, FIGURES_DIR / "pixel_contamination_scenarios.png", dpi=300)
    print(f"Saved: {FIGURES_DIR / 'pixel_contamination_scenarios.png'}")

    log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 9: Threshold Recommendation
# ============================================================

log.start_step("Threshold Recommendation")

recommended_eval = next(
    (e for e in threshold_evaluations if e["threshold"] == recommended_threshold),
    None,
)

print("THRESHOLD RECOMMENDATION")
print("=" * 60)
print(f"Recommended: {recommended_threshold}m")

for city in CITIES:
    retention = recommended_eval["per_city"][city]["retention_rate"]
    deviation = recommended_eval["per_city"][city]["max_deviation"]
    print(f"Retention ({city.title()}): {retention * 100:.1f}%")
    print(f"Genus deviation ({city.title()}): {deviation * 100:.2f}%")

print(f"Pixel coverage: {recommended_threshold / 10:.1f} pixels")
print(f"Meets retention target: {recommended_eval['passes_retention']}")
print(f"Genus uniformity: {recommended_eval['passes_uniformity']}")
print(f"Covers >=2 pixels: {recommended_eval['covers_two_pixels']}")

log.end_step(status="success")


# %%
# ============================================================
# SECTION 10: Export JSON Configuration
# ============================================================

log.start_step("Export Configuration")

impact_per_threshold = {}
for threshold in THRESHOLDS_TO_TEST:
    impact_per_threshold[str(threshold)] = {}
    for city in CITIES:
        row = sensitivity_results[
            (sensitivity_results["city"] == city)
            & (sensitivity_results["threshold"] == threshold)
        ].iloc[0]
        impact_per_threshold[str(threshold)][city] = {
            "trees_removed": int(row["trees_removed"]),
            "retention_rate": float(row["retention_rate"]),
        }

output = {
    "geometric_definition": {
        "distance_metric": "euclidean_point_to_point",
        "tree_geometry": "centroid (point)",
        "sentinel_pixel_size": "10m x 10m",
        "threshold_interpretation": "20m = approx 2-pixel separation (edge-to-edge contact)",
        "contamination_zones": {
            "high": "< 10m (full pixel overlap)",
            "medium": "10-15m (partial overlap)",
            "low": "15-20m (edge contact)",
            "none": "> 20m (no pixel overlap)",
        },
    },
    "threshold_sensitivity": {
        str(row["threshold_m"]): {
            "retention_rate": float(row["retention_rate"]),
            "removed_count": int(row["removed_count"]),
            "retained_count": int(row["retained_count"]),
        }
        for _, row in sensitivity_df.iterrows()
    },
    "version": "1.0",
    "created": datetime.now(timezone.utc).isoformat(),
    "recommended_threshold_m": int(recommended_threshold),
    "justification": (
        f"Threshold {recommended_threshold}m provides the best passing balance between "
        "spectral purity and retention, meeting retention and genus-uniformity criteria "
        "while covering at least two Sentinel-2 pixels."
    ),
    "thresholds_tested": THRESHOLDS_TO_TEST,
    "impact_per_threshold": impact_per_threshold,
    "genus_specific_uniform": bool(is_uniform.all()),
    "max_genus_deviation_percent": float(max_deviation.max() * 100),
    "distance_percentiles": stats_by_city,
    "validation": {
        "retention_exceeds_85_percent": bool(recommended_eval["passes_retention"]),
        "genus_impact_uniform": bool(recommended_eval["passes_uniformity"]),
        "covers_two_pixels": bool(recommended_eval["covers_two_pixels"]),
    },
}

json_path = METADATA_DIR / "proximity_filter.json"
json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"Saved: {json_path}")

log.end_step(status="success")


# %%
# ============================================================
# SECTION 11: Summary & Manual Sync Instructions
# ============================================================

# Save execution log
log.summary()
log_path = LOGS_DIR / f"{log.notebook}_execution.json"
log.save(log_path)
print(f"Execution log saved: {log_path}")

print("\n" + "=" * 60)
print("OUTPUT SUMMARY")
print("=" * 60)

print("\n--- JSON CONFIGURATIONS ---")
json_files = list(METADATA_DIR.glob("*.json"))
for f in sorted(json_files):
    print(f"  {f.name}")

print("\n--- PLOTS CREATED ---")
plot_files = list(FIGURES_DIR.glob("*.png"))
for f in sorted(plot_files):
    print(f"  {f.name}")

print(f"\nTotal plots: {len(plot_files)}")

print("\n" + "=" * 60)
print("⚠️  MANUAL SYNC REQUIRED")
print("=" * 60)
print("\nJSON configurations must be synced to Git repo:")
print("1. Download from Google Drive:")
for f in json_files:
    print(f"   - {f.relative_to(DRIVE_DIR)}")

print("\n2. Copy to local repo:")
print("   - Destination: outputs/phase_2/metadata/")

print("\n3. Commit and push to Git")
print("   - git add outputs/phase_2/metadata/*.json")
print("   - git commit -m 'Add proximity filter config'")
print("   - git push")

print("\n4. (Optional) Commit plots for documentation:")
print(f"   - Source: {FIGURES_DIR}")
print("   - Destination: outputs/phase_2/figures/exp_06_proximity/")

print("\n" + "=" * 60)
print("NOTEBOOK COMPLETE")
print("=" * 60)
