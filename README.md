# Urban Tree Transfer

Cross-City Transfer Learning for urban tree genus classification with Sentinel-2 satellite data.

## Research Question

How well do ML/DL models for tree genus classification transfer from Berlin to Leipzig, and how much local data is required for performance recovery?

## Tech Stack

- Python 3.10+
- UV (package management)
- Ruff (linting/formatting)
- Pyright (type checking)
- Nox (task automation)
- Pydantic (validation)
- GeoPandas, Rasterio (geospatial)
- scikit-learn, XGBoost, PyTorch (ML/DL)

## Setup

```bash
# Clone and install
git clone https://github.com/SilasPignotti/urban-tree-transfer.git
cd urban-tree-transfer
uv sync

# Verify installation
uv run nox -s ci
```

## Development Workflow

```bash
# Auto-fix linting issues before commit
uv run nox -s fix

# Run full CI pipeline
uv run nox -s ci
```

## Available Nox Sessions

```bash
uv run nox --list
```

## Colab Usage

```python
# In Google Colab
!pip install git+https://github.com/SilasPignotti/urban-tree-transfer.git -q

from google.colab import drive
drive.mount('/content/drive')

from urban_tree_transfer.config import get_data_dir
from urban_tree_transfer.feature_engineering import extraction
```

## Project Structure

```
urban-tree-transfer/
├── src/urban_tree_transfer/      # Source package
│   ├── config/                   # Paths, cities, features
│   ├── data_processing/          # Boundaries, Trees, Elevation, CHM, Sentinel
│   ├── feature_engineering/      # Extraction, QC, Selection, Splits
│   ├── experiments/              # Models, Training, Evaluation
│   └── utils/                    # IO, Visualization
├── configs/                      # YAML configurations
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
└── pyproject.toml
```

## Documentation

- [CLAUDE.md](CLAUDE.md) — Development guidelines
- [docs/PROJECT.md](docs/PROJECT.md) — Project requirements
