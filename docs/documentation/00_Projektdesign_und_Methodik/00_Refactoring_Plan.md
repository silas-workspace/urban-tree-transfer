# Refactoring-Plan

## Ausgangssituation

Dieses Repository ist ein Neustart des Projekts "Tree Species Classification". Das Legacy-Repo wird nicht weiter entwickelt, dient aber als Code-Referenz.

**Legacy-Repo:** `/Users/silas/Documents/projects/uni/Geo Projektarbeit/project`
**GitHub:** https://github.com/SilasPignotti/tree-classification

---

## Änderungen gegenüber Legacy

### Forschungsfrage (vereinfacht)
| Aspekt | Legacy | Neu |
|--------|--------|-----|
| Training | Berlin + Hamburg | Berlin |
| Transfer-Ziel | Rostock | Leipzig |
| Begründung | - | Keine Küstenklima-Confounds, klarere Methodik |

### Architektur (neu)
| Komponente | Legacy | Neu |
|------------|--------|-----|
| Code | Scripts + Notebooks gemischt | `src/` Package (installierbar) |
| Config | `scripts/config.py` monolithisch | `configs/` YAML-Dateien |
| Execution | Notebooks direkt | Runner-Notebooks importieren `src/` |
| Daten | Lokal + Drive gemischt | Drive (Daten), GitHub (Code + Metadaten) |

---

## Ziel-Verzeichnisstruktur

```
urban-tree-transfer/
├── src/                          # Installierbares Python-Package
│   ├── config/                   # Pfade, Städte, Features
│   ├── data_processing/          # Boundaries, Trees, Elevation, CHM, Sentinel
│   ├── feature_engineering/      # Extraction, QC, Selection, Splits
│   ├── experiments/              # Models, Training, Evaluation
│   └── utils/                    # IO, Visualization
├── configs/                      # YAML-Konfigurationen
│   ├── cities/                   # berlin.yaml, leipzig.yaml
│   └── experiments/              # phase_1.yaml, phase_2.yaml, phase_3.yaml
├── notebooks/
│   ├── runners/                  # Leichtgewichtige Colab-Runner
│   └── exploratory/              # Entwicklungs-Notebooks
├── docs/documentation/           # Methodikdokumentation
└── pyproject.toml
```

---

## Migration vom Legacy-Repo

### Zu adaptieren

| Legacy-Datei | Ziel | Anpassungen |
|--------------|------|-------------|
| `scripts/config.py` | `src/config/paths.py`, `cities.py` | Aufteilen, Hamburg/Rostock entfernen |
| `scripts/tree_cadastres/harmonize_tree_cadastres.py` | `src/data_processing/tree_cadastres.py` | Nur Berlin, Leipzig hinzufügen |
| `scripts/tree_cadastres/filter_trees.py` | `src/data_processing/tree_cadastres.py` | Integrieren |
| `scripts/elevation/*.py` | `src/data_processing/elevation.py` | Zusammenführen |
| `scripts/chm/*.py` | `src/data_processing/chm.py` | Zusammenführen |
| `notebooks/02_feature_engineering/*.ipynb` | `src/feature_engineering/` | Logik extrahieren |

### Phase 0 Ergebnisse (übernehmen)
Die Entscheidungen aus Phase 0 bleiben gültig:
- **CHM-Strategie:** Kein CHM (Overfitting-Risiko)
- **Dataset:** 20m-Edge (Gattungs-Isolations-Filter)
- **Features:** Top-50 nach JM-Distance

Pfad im Legacy-Repo: `data/03_experiments/00_phase_0/03_experiment_feature_reduction/metadata/selected_features.json`

### Nicht übernehmen
- Hamburg/Rostock-spezifischer Code
- Alte Notebook-Struktur
- Legacy-Dokumentation

---

## Wochenplan

### Woche 1: Code-Struktur

**Tag 1-2: Basis**
- [ ] `pyproject.toml` erstellen (Package installierbar)
- [ ] `src/__init__.py` Dateien anlegen
- [ ] `src/config/paths.py` - Environment Detection (local vs. Colab)
- [ ] `src/config/cities.py` - Stadt-Definitionen (Berlin, Leipzig)

**Tag 3-4: Data Processing Module**
- [ ] `src/data_processing/boundaries.py` - Stadtgrenzen
- [ ] `src/data_processing/tree_cadastres.py` - Download, Harmonisierung, Filter
- [ ] `src/data_processing/elevation.py` - DOM/DGM Processing
- [ ] `src/data_processing/chm.py` - CHM Erstellung + Resampling

**Tag 5: Runner-Template**
- [ ] `notebooks/runners/TEMPLATE.ipynb` - Colab-Runner Vorlage
- [ ] Test: Package Installation in Colab

### Woche 2: Leipzig-Daten

**Tag 6-7: Datenquellen verifizieren**
- [ ] Leipzig Baumkataster WFS testen
- [ ] Sachsen DOM/DGM Download identifizieren
- [ ] `harmonize_leipzig()` Funktion erstellen

**Tag 8-9: Daten prozessieren**
- [ ] Leipzig Boundary herunterladen
- [ ] Leipzig Baumkataster herunterladen + harmonisieren
- [ ] Leipzig DOM/DGM herunterladen + CHM erstellen

**Tag 10: Feature-Extraktion**
- [ ] Sentinel-2 Download für Leipzig (GEE)
- [ ] Feature-Extraktion ausführen
- [ ] Qualitätskontrolle

### Woche 3: Experimente

**Tag 11-12: Phase 1 - Berlin-Optimierung**
- [ ] Algorithmus-Vergleich (RF, XGBoost, 1D-CNN, TabNet)
- [ ] Best Model + Hyperparameter fixieren

**Tag 13-14: Phase 2 - Transfer**
- [ ] Zero-Shot Transfer Berlin → Leipzig
- [ ] Genus-spezifische Analyse

**Tag 15: Phase 3 - Fine-Tuning**
- [ ] Fine-Tuning-Kurve (0%, 10%, 25%, 50% Leipzig-Daten)

**Tag 16-17: Dokumentation**
- [ ] Ergebnisse dokumentieren
- [ ] Cleanup

---

## Datenquellen Leipzig

### Baumkataster
- **URL:** `https://geodienste.leipzig.de/l3/OpenData/Baeume/wfs`
- **Typ:** WFS
- **Status:** Zu verifizieren

### DOM/DGM
- **URL:** `https://www.geodaten.sachsen.de/downloadbereich-digitale-hoehenmodelle-4851.html`
- **Format:** XYZ in ZIP, 2km x 2km Tiles
- **CRS:** EPSG:25833 (→ transformieren nach 25832)
- **Jahr:** 2022

---

## Kritische Pfade

1. **Leipzig WFS muss funktionieren** (Tag 6)
   - Fallback: Open Data Portal direkter Download

2. **DOM/DGM Download** (Tag 8)
   - Fallback: Manueller Download der benötigten Tiles

3. **Referenzjahr-Harmonisierung**
   - Berlin: 2021
   - Leipzig: 2022
   - Sentinel-2 Daten pro Stadt an DOM/DGM-Jahr anpassen
