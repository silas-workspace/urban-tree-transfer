# Urban Tree Transfer

**Last Updated**: 2026-01-21
**Status**: ACTIVE
**Owner**: Silas Pignotti

---

## Overview

Cross-City Transfer Learning für urbane Baumgattungs-Klassifikation mit Sentinel-2 Satellitendaten.

**Problem**: ML-Modelle zur Baumklassifikation zeigen erhebliche Leistungseinbußen bei Anwendung auf neue Städte. Die benötigte Menge an lokalen Trainingsdaten für effektives Fine-Tuning ist unklar.

**Forschungsfrage**: Wie gut lassen sich ML/DL-Modelle zur Baumgattungs-Klassifikation von Berlin auf Leipzig übertragen, und wie viel lokale Daten sind für Performance-Recovery erforderlich?

---

## Objectives

- [ ] **Phase 1**: Optimales Modell für Single-City-Klassifikation (Berlin) finden
- [ ] **Phase 2**: Transfer-Performance quantifizieren (Berlin → Leipzig Zero-Shot)
- [ ] **Phase 3**: Fine-Tuning-Effizienz bestimmen (Datenmenge vs. Performance-Recovery)

---

## Architecture

### High-Level Structure

```
urban-tree-transfer/
├── src/                          # Python-Package (installierbar)
│   ├── config/                   # Pfade, Städte, Features
│   ├── data_processing/          # Boundaries, Trees, Elevation, CHM, Sentinel
│   ├── feature_engineering/      # Extraction, QC, Selection, Splits
│   ├── experiments/              # Models, Training, Evaluation
│   └── utils/                    # IO, Visualization
├── configs/                      # YAML-Konfigurationen
│   ├── cities/                   # berlin.yaml, leipzig.yaml
│   └── experiments/              # phase_1.yaml, phase_2.yaml, phase_3.yaml
├── notebooks/
│   ├── runners/                  # Colab-Runner (importieren src/)
│   └── exploratory/              # Entwicklungs-Notebooks
├── docs/documentation/           # Methodikdokumentation
└── pyproject.toml
```

### Key Components

| Component               | Purpose                                  | Status   |
| ----------------------- | ---------------------------------------- | -------- |
| **Data Processing**     | Download, Harmonisierung, CHM-Erstellung | Planning |
| **Feature Engineering** | Extraktion, QC, Selection, Splits        | Planning |
| **Experiments**         | Model Training, Evaluation, Transfer     | Planning |

### Dependencies

**External Services**:

- Google Earth Engine: Sentinel-2 Download
- Google Drive: Datenspeicher
- Google Colab: GPU-Ausführung

**Data Sources**:

- Berlin Baumkataster: WFS via gdi.berlin.de
- Leipzig Baumkataster: WFS via geodienste.leipzig.de
- DOM/DGM: Landesvermessung Berlin/Sachsen

---

## Getting Started

### Prerequisites

- Python 3.10+
- uv (Package Manager)
- Google Account (für Colab/Drive)

### Installation

```bash
git clone https://github.com/SilasPignotti/urban-tree-transfer.git
cd urban-tree-transfer
uv sync
```

### Colab Usage

```python
# In Google Colab
!pip install git+https://github.com/SilasPignotti/urban-tree-transfer.git -q

from google.colab import drive
drive.mount('/content/drive')

from src.config import get_data_dir
from src.feature_engineering import extraction
```

---

## Current Status

### Completed

- [x] Projektdesign und Forschungsfrage definiert
- [x] Methodische Grundlagen dokumentiert
- [x] Verzeichnisstruktur erstellt

### In Progress

- [ ] Code-Struktur aufbauen (src/ Package)
- [ ] Leipzig-Datenquellen verifizieren

### Planned

- [ ] Phase 1: Berlin-Optimierung
- [ ] Phase 2: Transfer-Evaluation
- [ ] Phase 3: Fine-Tuning-Analyse

---

## Study Design

### Cities

| Stadt       | Rolle             | Daten                      |
| ----------- | ----------------- | -------------------------- |
| **Berlin**  | Training (Source) | ~800k Bäume, DOM/DGM 2021  |
| **Leipzig** | Transfer-Ziel     | Baumkataster, DOM/DGM 2022 |

### Methods

| Methode           | Typ | Beschreibung                         |
| ----------------- | --- | ------------------------------------ |
| **Random Forest** | ML  | Baseline, interpretierbar            |
| **XGBoost**       | ML  | Gradient Boosting                    |
| **1D-CNN**        | DL  | Temporale Convolution auf Features   |
| **TabNet**        | DL  | Attention-basiert für tabulare Daten |

### Features

- **Spektral**: 10 Sentinel-2 Bänder (B02-B12)
- **Indices**: NDVI, EVI, Red-Edge, Wasser-Indices
- **Temporal**: Monatliche Komposite (April-November)
- **Strukturell**: CHM (optional, mit Vorsicht)

---

## Documentation

- [Projektübersicht](documentation/00_Projektdesign_und_Methodik/01_Projektuebersicht.md)
- [Methodische Grundlagen](documentation/00_Projektdesign_und_Methodik/02_Methodische_Grundlagen.md)
- [Refactoring Plan](documentation/00_Projektdesign_und_Methodik/00_Refactoring_Plan.md)

---

## Technical Details

### Coordinate Systems

- **Berlin Source**: EPSG:25833
- **Leipzig Source**: EPSG:25833

### Data Formats

- **Spatial**: GeoPackage (.gpkg)
- **ML-Ready**: Parquet (.parquet)
- **Config**: YAML (.yaml)
- **Metadata**: JSON (.json)

### Random Seed

`42` für alle stochastischen Prozesse

---

## Related Resources

- **Legacy Repo**: [tree-classification](https://github.com/SilasPignotti/tree-classification)

---

**Last Reviewed**: 2026-01-21
