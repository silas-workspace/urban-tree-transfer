# Google Drive Folder Structure

**Last Updated**: 2026-02-04
**Purpose**: Standardized folder structure for all project phases

---

## 📁 Complete Structure

```
/content/drive/MyDrive/dev/urban-tree-transfer/
│
├── data/                                    # Input data only
│   ├── phase_1_processing/                 # Phase 1 data files
│   │   ├── trees/
│   │   │   └── trees_filtered_viable.gpkg  # Combined Berlin + Leipzig trees
│   │   ├── chm/
│   │   │   ├── CHM_1m_berlin.tif          # Berlin canopy height model
│   │   │   └── CHM_1m_leipzig.tif         # Leipzig canopy height model
│   │   └── sentinel2/
│   │       ├── S2_berlin_2021_01.tif      # Berlin S2 composites (12 months)
│   │       ├── S2_berlin_2021_02.tif
│   │       ├── ...
│   │       ├── S2_leipzig_2021_01.tif     # Leipzig S2 composites (12 months)
│   │       ├── S2_leipzig_2021_02.tif
│   │       └── ...
│   │
│   ├── phase_2_features/                   # Phase 2 data files
│   │   ├── trees_with_features_berlin.gpkg # From 02a (with all temporal features)
│   │   ├── trees_with_features_leipzig.gpkg
│   │   ├── trees_clean_berlin.gpkg         # From 02b (quality-controlled, 0 NaN)
│   │   └── trees_clean_leipzig.gpkg
│   │
│   └── phase_2_splits/                     # Phase 2c data files (10 GeoPackages)
│       ├── berlin_train.gpkg               # Baseline splits (5 files)
│       ├── berlin_val.gpkg
│       ├── berlin_test.gpkg
│       ├── leipzig_finetune.gpkg
│       ├── leipzig_test.gpkg
│       ├── berlin_train_filtered.gpkg      # Filtered splits (5 files)
│       ├── berlin_val_filtered.gpkg
│       ├── berlin_test_filtered.gpkg
│       ├── leipzig_finetune_filtered.gpkg
│       └── leipzig_test_filtered.gpkg
│
├── outputs/                                 # All outputs (metadata, logs, figures)
│   ├── phase_1/                            # Phase 1 outputs
│   │   ├── metadata/                       # Phase 1 metadata
│   │   │   ├── trees_cadastre_summary.json
│   │   │   ├── sentinel2_tasks.json
│   │   │   └── ...
│   │   └── logs/                           # Phase 1 execution logs
│   │       └── 01_data_processing_execution.json
│   │
│   ├── phase_2/                            # Phase 2 outputs
│   │   ├── metadata/                       # Phase 2 metadata & exploratory JSONs
│   │   │   ├── temporal_selection.json     # From exp_01
│   │   │   ├── chm_assessment.json         # From exp_02
│   │   │   ├── correlation_removal.json    # From exp_03
│   │   │   ├── outlier_thresholds.json     # From exp_04
│   │   │   ├── spatial_autocorrelation.json # From exp_05
│   │   │   ├── proximity_filter.json       # From exp_06
│   │   │   ├── feature_extraction_summary.json
│   │   │   ├── feature_extraction_validation.json
│   │   │   ├── data_quality_summary.json
│   │   │   └── data_quality_validation.json
│   │   ├── logs/                           # Phase 2 execution logs
│   │   │   ├── 02a_feature_extraction_execution.json
│   │   │   ├── 02b_data_quality_execution.json
│   │   │   └── 02b_execution.log
│   │   └── figures/                        # Phase 2 visualizations
│   │       ├── exp_01_temporal/
│   │       │   ├── jm_distance_by_month_berlin.png
│   │       │   ├── jm_distance_by_month_leipzig.png
│   │       │   └── jm_rank_consistency.png
│   │       ├── exp_02_chm/
│   │       ├── exp_03_correlation/
│   │       ├── exp_04_outlier_thresholds/
│   │       ├── exp_05_spatial/
│   │       └── exp_06_proximity/
│   │
│   └── phase_2_splits/                     # Phase 2c outputs
│       ├── metadata/                       # Phase 2c metadata
│       │   └── phase_2_final_summary.json
│       ├── logs/                           # Phase 2c execution logs
│       │   └── 02c_final_preparation_execution.json
│       └── figures/                        # Phase 2c visualizations
│           └── 02c_final_prep/
│               ├── split_size_comparison.png
│               └── genus_distribution_comparison.png
│
└── [models/]                               # Phase 3 (future)
    └── [experiments/]
```

---

## 🎯 Design Principles

### **Separation of Data and Outputs**
- `data/` contains only data files (GeoPackages, GeoTIFFs) - large binary files
- `outputs/` contains metadata, logs, and figures - small, version-controllable files

### **Self-Contained Phases**
Each phase has two directories:
- `data/phase_N/` - Data files (GeoPackages, GeoTIFFs)
- `outputs/phase_N/` - Metadata (JSON), logs, figures (PNG)

### **Idempotent Pipelines**
All runner notebooks check if outputs exist and skip processing if found. This allows safe re-runs without data loss.

---

## 📊 Data Flow

```
Phase 1 Processing
    ├── trees_filtered_viable.gpkg (input for 02a)
    ├── CHM_1m_*.tif (input for 02a)
    └── S2_*_*.tif (input for 02a)
        ↓
Phase 2a: Feature Extraction (02a)
    └── trees_with_features_*.gpkg
        ↓
Exploratory Set 1 (exp_01, exp_02)
    └── temporal_selection.json, chm_assessment.json
        ↓
Phase 2b: Data Quality (02b)
    └── trees_clean_*.gpkg
        ↓
Exploratory Set 2 (exp_03, exp_04, exp_05, exp_06)
    └── 4 JSON configs
        ↓
Phase 2c: Final Preparation (02c)
    └── 10 split GeoPackages (baseline + filtered)
        ↓
Phase 3: Experiments (future)
```

---

## 📝 File Naming Conventions

### GeoPackages
- Input trees: `trees_filtered_viable.gpkg` (combined cities)
- Features: `trees_with_features_{city}.gpkg` (after 02a)
- Clean: `trees_clean_{city}.gpkg` (after 02b)
- Splits: `{city}_{split}.gpkg` or `{city}_{split}_filtered.gpkg` (after 02c)

### Rasters
- CHM: `CHM_1m_{city}.tif` (1m resolution)
- Sentinel-2: `S2_{city}_{year}_{month:02d}.tif` (10m resolution, monthly composites)

### Metadata
- Exploratory: `{analysis_name}.json` (e.g., `temporal_selection.json`)
- Summary: `{notebook}_summary.json` (e.g., `feature_extraction_summary.json`)
- Validation: `{notebook}_validation.json`

### Logs
- JSON: `{notebook}_execution.json` (structured)
- Text: `{notebook}_execution.log` (plain text)

### Figures
- Directory: `{notebook_or_exp}/` (e.g., `exp_01_temporal/`)
- Files: `{description}.png` (e.g., `jm_distance_by_month_berlin.png`)

---

## 🔧 Folder Creation

### Automatic
All runner and exploratory notebooks automatically create required subdirectories:
```python
for d in [OUTPUT_DIR, METADATA_DIR, LOGS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
```

### Manual (Optional)
You can pre-create the base structure if desired:
```bash
mkdir -p ~/MyDrive/dev/urban-tree-transfer/data/{phase_1_processing,phase_2_features,phase_2_splits}
```

---

## ⚠️ Important Notes

### DO NOT Commit to Git
- ✅ Commit: Metadata JSON files (small, version-controlled)
- ✅ Commit: Logs (for debugging)
- ❌ Do NOT commit: GeoPackage files (too large, stay in Drive)
- ❌ Do NOT commit: GeoTIFF files (too large, stay in Drive)
- ⚠️ Optional: Commit figures (PNG files, moderate size)

### Download for Git
After completing phases, download these from Drive to commit:
```bash
# Phase 1
outputs/phase_1/metadata/*.json
outputs/phase_1/logs/*.json

# Phase 2
outputs/phase_2/metadata/*.json
outputs/phase_2/logs/*.json
outputs/phase_2_splits/metadata/*.json
outputs/phase_2_splits/logs/*.json

# Optional: figures
outputs/phase_2/figures/**/*.png
outputs/phase_2_splits/figures/**/*.png
```

### Backup Strategy
- **Primary**: All data in Google Drive (auto-synced)
- **Secondary**: GeoPackages can be downloaded locally if needed
- **Version Control**: Only metadata/logs/notebooks in Git

---

## 📚 Related Documentation

- [README.md](../README.md) - Project overview
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [docs/PROJECT.md](PROJECT.md) - Project design
- [notebooks/README.md](../notebooks/README.md) - Notebook execution guide

---

**Consistency Check**: If you find paths that don't match this structure, update them to maintain consistency across all phases.
