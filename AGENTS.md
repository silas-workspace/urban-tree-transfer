# Urban Tree Transfer

## Project Overview

Urban Tree Transfer — genus-level classification of urban trees from multitemporal
Sentinel-2 imagery. Cross-city transfer learning study (Berlin -> Leipzig).
See `docs/PROJECT.md` for full project description, study design, and pipeline details.

**Status**: Active rework. The codebase is functional and has produced results.
We are refactoring for quality, not building from scratch.

**Owner**: Silas Pignotti
**Repository**: github.com/silas-workspace/urban-tree-transfer

---

## Critical Rules

### 1. Handle With Care
This is a **research codebase with existing results**. Every change must preserve
the ability to reproduce current outputs. Do not refactor speculatively.

- **No bulk rewrites.** Change only what the task explicitly asks for.
- **No silent behavior changes.** If a function's output would change, flag it.
- **No dependency upgrades** unless explicitly requested.
- **Preserve random seeds** (RANDOM_SEED = 42) everywhere.

### 2. Execution Environment
Code runs in **Google Colab** notebooks, NOT locally.

- You **cannot** run notebooks or execute the pipeline.
- You **can** edit Python source files under `src/` and notebook `.ipynb` files.
- You **can** run linting/typechecks if configured, but NOT the actual pipeline.
- All data lives on **Google Drive** (~24 GB), not in the repo.

### 3. Change Scope
Before making any change, verify:
- [ ] Does the task explicitly ask for this change?
- [ ] Could this affect downstream outputs?
- [ ] Is the change isolated to the specified file(s)?

If unsure, ask before proceeding.

---

## Architecture

```
urban-tree-transfer/
├── src/urban_tree_transfer/      # Installable Python package
│   ├── config/                   # Config loading, constants, experiment helpers
│   ├── configs/                  # YAML: cities/, experiments/, features/
│   ├── data_processing/          # Boundaries, trees, elevation, CHM, Sentinel-2
│   ├── feature_engineering/      # Extraction, quality, outliers, proximity, splits
│   ├── experiments/              # Models, training, ablation, transfer, evaluation
│   ├── schemas/                  # JSON schemas for pipeline output validation
│   └── utils/                    # IO, plotting, logging, validation helpers
├── notebooks/
│   ├── runners/                  # Colab runner notebooks (01_, 02a-c_, 03a-d_)
│   ├── exploratory/              # Experiment notebooks (exp_01 to exp_11)
│   └── templates/                # Notebook template
├── tests/                        # Unit + integration tests (mirrors src/ structure)
├── docs/                         # PROJECT.md, methodology docs, PRDs
├── scripts/                      # Standalone utility scripts
├── outputs/                      # Execution logs and metadata (committed)
└── legacy/                       # Superseded code — do not import from here
```

### Package Install (Colab)
```
pip install git+https://{token}@github.com/silas-workspace/urban-tree-transfer.git
```
All shared logic lives in `src/`. Notebooks only orchestrate — they import from
the package and handle I/O with Google Drive.

### Notebook Types
Two types, each following a strict template:

| Type | Purpose | Pattern | Output |
|------|---------|---------|--------|
| **Runner** | Data processing | Load -> Process -> Save -> Validate | Parquet/GeoPackage files |
| **Exploratory** | Analysis & decisions | Objective -> Method -> Results -> Interpretation | JSON configs + CSVs |

### Key Libraries
Python 3.10+, GeoPandas, rasterio, scikit-learn, XGBoost, PyTorch, Optuna, scipy

### Data Formats
| Type | Format |
|------|--------|
| Spatial | GeoPackage (`.gpkg`) |
| ML-ready | Parquet (`.parquet`, Snappy, no geometry) |
| Config | YAML (`.yaml`) |
| Metadata | JSON (`.json`, schema-validated) |

---

## Code Quality

- **Line length:** 100 characters
- **Quotes:** double
- **Type hints:** required on all function signatures
- **Docstrings:** Google style for all public functions
- **CRS:** EPSG:25833 (UTM zone 33N) — always reproject before any spatial op
- **File paths:** always `pathlib.Path`, never raw strings

### Nox Sessions
```bash
uv run nox -s lint        # ruff check
uv run nox -s format      # ruff format
uv run nox -s typecheck   # pyright
uv run nox -s fix         # auto-fix lint + format
uv run nox -s test        # pytest (unit only, no integration)
uv run nox -s pre_commit  # fix + typecheck — run before every commit
uv run nox -s ci          # full pipeline
```

---

## Notebook Style

- Structured header cell with metadata (title, phase, purpose, I/O)
- Numbered sections with clear separators
- Each section: purpose comment -> code -> status output
- Final cell: execution summary + output manifest
- No visualization code in notebooks (visualizations are built separately)

---

## Communication

- If a task is ambiguous, ask for clarification before changing code.
- If a change would affect more files than specified, list them and confirm.
- After each task, summarize: files changed, what changed, what to verify in Colab.
