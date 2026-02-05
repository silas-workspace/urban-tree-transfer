# PRD: Phase 2c Extension — Parquet Export & Geometry Lookup

**PRD ID:** 002c-ext
**Status:** Draft
**Created:** 2026-02-05
**Last Updated:** 2026-02-05
**Parent PRD:** 002_phase2_feature_engineering_overview.md

---

## 1. Overview

### 1.1 Problem Statement

Phase 2c currently outputs 10 GeoPackage files (`.gpkg`) as the final ML-ready datasets. While GeoPackages are excellent for geospatial data integrity, they introduce significant overhead for ML training:

- **Slow I/O:** GeoPackage uses SQLite internally — loading a 200-column GeoDataFrame is 5-10× slower than Parquet
- **Unnecessary geometry:** ML training uses only tabular features, not point coordinates
- **Heavy dependencies:** `geopandas` + `fiona` required even when geometry is unused
- **Column selection:** GeoPackage reads all columns; Parquet supports selective column loading
- **Colab impact:** On Google Colab, the slower load times compound across multiple notebooks (exp_08, exp_09, exp_10, 03a-03d each load multiple splits)

### 1.2 Goals

1. **Add Parquet export** alongside existing GeoPackage output in Phase 2c runner
2. **Separate geometry** into a lightweight lookup file for later visualization
3. **Zero breaking changes** — GeoPackages remain as authoritative Phase 2 output
4. **Phase 3 efficiency** — All Phase 3 notebooks load Parquet instead of GeoPackage

### 1.3 Non-Goals

- Replacing GeoPackages (they remain for traceability and spatial debugging)
- Changing Phase 2c processing logic (splits, outliers, etc. stay identical)
- Optimizing Phase 1 or Phase 2a/2b data formats

---

## 2. Design

### 2.1 Output Structure

After this extension, Phase 2c produces:

```
data/phase_2_splits/
├── # EXISTING: GeoPackages (authoritative, unchanged)
├── berlin_train.gpkg
├── berlin_val.gpkg
├── berlin_test.gpkg
├── leipzig_finetune.gpkg
├── leipzig_test.gpkg
├── berlin_train_filtered.gpkg
├── berlin_val_filtered.gpkg
├── berlin_test_filtered.gpkg
├── leipzig_finetune_filtered.gpkg
├── leipzig_test_filtered.gpkg
│
├── # NEW: Parquet files (ML-optimized, no geometry)
├── berlin_train.parquet
├── berlin_val.parquet
├── berlin_test.parquet
├── leipzig_finetune.parquet
├── leipzig_test.parquet
├── berlin_train_filtered.parquet
├── berlin_val_filtered.parquet
├── berlin_test_filtered.parquet
├── leipzig_finetune_filtered.parquet
├── leipzig_test_filtered.parquet
│
└── # NEW: Geometry lookup (for visualization)
    └── geometry_lookup.parquet
```

**Total:** 10 GeoPackages (existing) + 10 Parquet (new) + 1 geometry lookup (new)

### 2.2 Parquet Schema

Each Parquet file contains all columns from the corresponding GeoPackage **except** `geometry`:

#### Retained Columns

| Category           | Columns                                                                                            | Purpose                       |
| ------------------ | -------------------------------------------------------------------------------------------------- | ----------------------------- |
| **Join Key**       | `tree_id`                                                                                          | Link back to geometry lookup  |
| **Label**          | `genus_latin`                                                                                      | Classification target         |
| **City**           | `city`                                                                                             | City identifier for filtering |
| **ML Features**    | All CHM + S2 temporal features                                                                     | Model input                   |
| **Outlier Flags**  | `outlier_zscore`, `outlier_mahalanobis`, `outlier_iqr`, `outlier_severity`, `outlier_method_count` | Ablation study filters        |
| **Split Metadata** | `block_id`                                                                                         | Spatial CV block assignment   |

#### Dropped Columns

| Column                | Reason                                       |
| --------------------- | -------------------------------------------- |
| `geometry`            | Not needed for training; available in lookup |
| `species_latin`       | Not used as feature or label                 |
| `genus_german`        | Not used as feature or label                 |
| `species_german`      | Not used as feature or label                 |
| `plant_year`          | Metadata only, not an ML feature             |
| `height_m`            | Redundant with CHM features                  |
| `tree_type`           | Metadata only                                |
| `position_corrected`  | QC metadata only                             |
| `correction_distance` | QC metadata only                             |

**Rationale:** Parquet files should contain only what Phase 3 actually uses. Metadata columns add I/O overhead and can always be joined back via `tree_id` from the GeoPackage if needed.

### 2.3 Geometry Lookup Schema

A single Parquet file containing the spatial reference for all trees across all splits:

```
geometry_lookup.parquet
├── tree_id (string, non-null) — primary key
├── city (string, non-null)
├── split (string, non-null) — e.g. "berlin_train", "leipzig_test"
├── filtered (boolean, non-null) — whether tree is in filtered variant
├── x (float64, non-null) — UTM easting (EPSG:25833)
└── y (float64, non-null) — UTM northing (EPSG:25833)
```

**Why Parquet, not GeoPackage?**

- This lookup serves Phase 3 visualization (plotting predictions on a map)
- Storing x/y as floats is simpler and faster than a GeoDataFrame
- Can be joined with prediction results: `pd.merge(predictions, geo_lookup, on="tree_id")`
- Reconstructing geometry is trivial: `gpd.points_from_xy(df.x, df.y, crs="EPSG:25833")`

### 2.4 Data Flow

```
Phase 2c Runner (02c_final_preparation.ipynb)
│
├── [EXISTING] Create splits, outlier flags, proximity filter
│
├── [EXISTING] Save 10 GeoPackages
│          └── .gpkg files (full schema, authoritative)
│
├── [NEW] Export 10 Parquet files
│          └── Drop geometry + metadata columns
│          └── Save with Snappy compression
│
└── [NEW] Export geometry lookup
           └── Extract tree_id, city, split, filtered, x, y
           └── Deduplicate (tree in baseline + filtered → one row, filtered=True)
           └── Save as Parquet
```

---

## 3. Implementation

### 3.1 New Utility Function

Add to `src/urban_tree_transfer/feature_engineering/selection.py`:

```python
def export_splits_to_parquet(
    splits: dict[str, gpd.GeoDataFrame],
    output_dir: Path,
    drop_metadata_columns: list[str] | None = None,
) -> dict[str, Path]:
    """Export GeoDataFrame splits to Parquet files (without geometry).

    Drops geometry and specified metadata columns, then saves each split
    as a Parquet file with Snappy compression.

    Parameters
    ----------
    splits
        Mapping of split name (e.g. "berlin_train") to GeoDataFrame.
    output_dir
        Directory to write Parquet files into.
    drop_metadata_columns
        Additional columns to drop beyond geometry. If None, uses default
        list of non-ML columns (species, plant_year, height_m, etc.).

    Returns
    -------
    dict[str, Path]
        Mapping of split name to output Parquet path.
    """
```

```python
def export_geometry_lookup(
    splits: dict[str, gpd.GeoDataFrame],
    output_path: Path,
) -> int:
    """Export geometry lookup table for all trees across splits.

    Creates a single Parquet file with tree_id, city, split, x, y
    for later visualization of predictions on maps.

    Parameters
    ----------
    splits
        Mapping of split name to GeoDataFrame. Names must follow pattern
        "{city}_{split}" or "{city}_{split}_filtered".
    output_path
        Path for the output Parquet file.

    Returns
    -------
    int
        Number of unique trees in the lookup.
    """
```

### 3.2 Default Metadata Columns to Drop

```python
_PARQUET_DROP_COLUMNS = [
    "geometry",
    "species_latin",
    "genus_german",
    "species_german",
    "plant_year",
    "height_m",
    "tree_type",
    "position_corrected",
    "correction_distance",
]
```

### 3.3 Notebook Changes (02c_final_preparation.ipynb)

Add a new cell block **after** the existing GeoPackage export cell:

```python
# ============================================================================
# PARQUET EXPORT (ML-optimized format for Phase 3)
# ============================================================================
from urban_tree_transfer.feature_engineering.selection import (
    export_splits_to_parquet,
    export_geometry_lookup,
)

# Collect all splits (baseline + filtered)
all_splits = {}
for city, splits in split_results.items():
    for split_name, split_gdf in splits["baseline"].items():
        all_splits[f"{city}_{split_name}"] = split_gdf
    for split_name, split_gdf in splits["filtered"].items():
        all_splits[f"{city}_{split_name}_filtered"] = split_gdf

# Export Parquet files
parquet_paths = export_splits_to_parquet(all_splits, OUTPUT_DIR)
print(f"Exported {len(parquet_paths)} Parquet files")

# Export geometry lookup
n_trees = export_geometry_lookup(all_splits, OUTPUT_DIR / "geometry_lookup.parquet")
print(f"Geometry lookup: {n_trees} unique trees")
```

### 3.4 Compression & Performance

| Setting     | Value     | Rationale                                       |
| ----------- | --------- | ----------------------------------------------- |
| Engine      | `pyarrow` | Standard, fastest for read/write                |
| Compression | `snappy`  | Best read speed; slight size trade-off vs. gzip |
| Row groups  | Default   | Sufficient for datasets < 1M rows               |

**Expected size savings** (estimated for ~30k trees × 200 features):

| Format           | Estimated Size      |
| ---------------- | ------------------- |
| GeoPackage       | ~50-80 MB per split |
| Parquet (Snappy) | ~5-15 MB per split  |
| Geometry Lookup  | ~1 MB total         |

**Expected load time improvement** (Google Colab, standard runtime):

| Operation             | GeoPackage    | Parquet   | Speedup |
| --------------------- | ------------- | --------- | ------- |
| Load full dataset     | ~5-10s        | ~0.5-1s   | ~10×    |
| Load selected columns | Not supported | ~0.2-0.5s | ~20×    |

---

## 4. Validation

### 4.1 Automated Checks (in notebook)

```python
# For each split: verify Parquet matches GeoPackage content
for name, gdf in all_splits.items():
    pq = pd.read_parquet(OUTPUT_DIR / f"{name}.parquet")

    # Row count must match
    assert len(pq) == len(gdf), f"{name}: row count mismatch"

    # Feature columns must match
    feature_cols = [c for c in pq.columns if c not in ["tree_id", "city", "genus_latin",
                    "block_id", "outlier_zscore", "outlier_mahalanobis",
                    "outlier_iqr", "outlier_severity", "outlier_method_count"]]
    for col in feature_cols:
        pd.testing.assert_series_equal(
            pq[col].reset_index(drop=True),
            gdf[col].reset_index(drop=True),
            check_names=False,
        )

    # Geometry must NOT be present
    assert "geometry" not in pq.columns

# Geometry lookup: all tree_ids recoverable
lookup = pd.read_parquet(OUTPUT_DIR / "geometry_lookup.parquet")
for name, gdf in all_splits.items():
    split_ids = set(gdf["tree_id"])
    lookup_ids = set(lookup[lookup["split"] == name]["tree_id"])
    assert split_ids == lookup_ids, f"{name}: tree_id mismatch in lookup"
```

### 4.2 Unit Tests

Add to `tests/feature_engineering/test_selection.py`:

```python
def test_export_splits_to_parquet_drops_geometry(tmp_path):
    """Parquet output must not contain geometry column."""

def test_export_splits_to_parquet_drops_metadata(tmp_path):
    """Parquet output must not contain non-ML metadata columns."""

def test_export_splits_to_parquet_preserves_features(tmp_path):
    """All ML feature values must be identical between GeoPackage and Parquet."""

def test_export_splits_to_parquet_preserves_row_count(tmp_path):
    """Row count must match between GeoPackage and Parquet."""

def test_export_geometry_lookup_schema(tmp_path):
    """Geometry lookup must have expected columns and no duplicates per split."""

def test_export_geometry_lookup_coordinates(tmp_path):
    """X/Y coordinates must match original geometry."""
```

---

## 5. Impact on Phase 3

### 5.1 Changes Required in Phase 3 PRD

All Phase 3 notebooks should load **Parquet** instead of GeoPackage:

| Phase 3 Component           | Before                               | After                                        |
| --------------------------- | ------------------------------------ | -------------------------------------------- |
| exp_08 CHM ablation         | `gpd.read_file("berlin_train.gpkg")` | `pd.read_parquet("berlin_train.parquet")`    |
| exp_09 Feature reduction    | `gpd.read_file("berlin_train.gpkg")` | `pd.read_parquet("berlin_train.parquet")`    |
| 03a Setup fixation          | `gpd.read_file("*.gpkg")`            | `pd.read_parquet("*.parquet")`               |
| exp_10 Algorithm comparison | Loads from 03a output                | Already Parquet (no change)                  |
| 03b-03d                     | Loads from 03a output                | Already Parquet (no change)                  |
| Visualization (optional)    | N/A                                  | `pd.read_parquet("geometry_lookup.parquet")` |

### 5.2 Phase 3 Data Flow (Updated)

```
Phase 2 Outputs
├── *.gpkg (authoritative, for traceability)
├── *.parquet (ML-optimized, for training)     ← NEW
└── geometry_lookup.parquet (for visualization) ← NEW
                    │
                    ▼ (Phase 3 loads .parquet)
          exp_08, exp_09, 03a
                    │
                    ▼
          03a exports to data/phase_3_experiments/*.parquet
                    │
                    ▼
          exp_10, 03b, 03c, 03d
                    │
                    ▼
          Visualization joins predictions with geometry_lookup.parquet
```

### 5.3 Colab Drive Structure (Updated)

```
Google Drive/
└── dev/urban-tree-transfer/
    ├── data/
    │   ├── phase_2_splits/                    # Phase 2c output
    │   │   ├── berlin_train.gpkg              # Authoritative (keep)
    │   │   ├── berlin_train.parquet           # ML-optimized (NEW)
    │   │   ├── berlin_val.parquet             # ML-optimized (NEW)
    │   │   ├── berlin_test.parquet            # ML-optimized (NEW)
    │   │   ├── leipzig_finetune.parquet       # ML-optimized (NEW)
    │   │   ├── leipzig_test.parquet           # ML-optimized (NEW)
    │   │   ├── *_filtered.parquet             # Filtered variants (NEW)
    │   │   ├── geometry_lookup.parquet        # Geo reference (NEW)
    │   │   └── ... (.gpkg files as before)
    │   └── phase_3_experiments/               # Phase 3 processed
    │       └── *.parquet
```

---

## 6. Success Criteria

- [ ] 10 Parquet files generated alongside 10 GeoPackages
- [ ] Geometry lookup contains all unique trees with correct coordinates
- [ ] Parquet row counts match GeoPackage row counts
- [ ] Feature values are identical between Parquet and GeoPackage
- [ ] No geometry column in Parquet files
- [ ] No unused metadata columns in Parquet files
- [ ] Unit tests pass for export functions
- [ ] Phase 3 PRD updated to reference Parquet inputs
- [ ] Phase 3 Experiment docs reference Parquet loading pattern

---

## 7. Timeline Estimate

| Task                                   | Estimated Hours |
| -------------------------------------- | --------------- |
| Implement `export_splits_to_parquet()` | 0.5h            |
| Implement `export_geometry_lookup()`   | 0.5h            |
| Update 02c notebook                    | 0.5h            |
| Unit tests                             | 0.5h            |
| Update Phase 3 PRD                     | 0.5h            |
| Update Schema documentation            | 0.5h            |
| **Total**                              | **~3h**         |

---

**Status:** Ready for Implementation
