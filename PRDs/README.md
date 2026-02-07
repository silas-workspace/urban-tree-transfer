# Product Requirements Documents (PRDs)

This directory contains all Product Requirements Documents for the urban-tree-transfer project.

## Active PRDs

### Phase 3: Experiments (Current)

Phase 3 is split into **4 sequential PRDs** based on dependencies:

- **[003a_setup.md](003a_setup.md)** - Setup-Fixierung (exp_08, exp_08b, exp_08c, exp_09)
  - CHM ablation, proximity filtering, outlier removal, feature reduction
  - Output: `setup_decisions.json`
  - Dependencies: Phase 2 outputs only

- **[003b_berlin.md](003b_berlin.md)** - Berlin-Optimierung (03a, exp_10, 03b)
  - Algorithm comparison (with naive baselines), HP-tuning, error analysis
  - Output: `algorithm_comparison.json`, trained champions
  - Dependencies: Requires `setup_decisions.json` from 003a

- **[003c_transfer.md](003c_transfer.md)** - Transfer-Evaluation (03c)
  - Zero-shot transfer, feature stability, a-priori hypothesis testing
  - Output: `transfer_evaluation.json`
  - Dependencies: Requires trained champions from 003b

- **[003d_finetuning.md](003d_finetuning.md)** - Fine-Tuning (03d)
  - Sample efficiency curves, power-law modeling, from-scratch comparison
  - Output: `finetuning_curve.json`
  - Dependencies: Requires trained champions from 003b

**Reference:** [003_phase3_complete.md](003_phase3_complete.md) - Complete unified PRD (for reference only)

**Execution Order:** 003a → 003b → (003c + 003d in parallel)

**Working with Coding Agents:** Provide the specific PRD for the current implementation step.

## Completed PRDs

### Phase 2: Feature Engineering

- **[002_phase2_feature_engineering_overview.md](002_phase2_feature_engineering_overview.md)** - ✅ Complete
- **[002_phase2/](002_phase2/)** - Modular task PRDs
  - [002a_feature_extraction.md](002_phase2/002a_feature_extraction.md) - CHM/S2 feature extraction
  - [002b_data_quality.md](002_phase2/002b_data_quality.md) - NaN handling, plausibility filters
  - [002c_final_preparation.md](002_phase2/002c_final_preparation.md) - Outliers, spatial splits
  - [002_exploratory.md](002_phase2/002_exploratory.md) - Exploratory analysis notebooks

### Phase 1: Data Processing

- **[done/001_phase1_data_processing.md](done/001_phase1_data_processing.md)** - ✅ Complete
  - Tree cadastre harmonization
  - CHM generation from DOM/DGM
  - Sentinel-2 monthly composites

## PRD Structure Philosophy

### Monolithic (Phase 1)

- Single comprehensive document
- Good for smaller, focused tasks
- Example: PRD 001

### Modular (Phase 2+)

- Main overview PRD + separate task PRDs
- Better for complex, multi-step work
- Easier for coding agent delegation
- Example: PRD 002 (overview + 002a/b/c)

## Templates

- **[templates/prd_simple.md](templates/prd_simple.md)** - Simple PRD template for small tasks

## Naming Convention

- `{NNN}_phase{N}_{topic}.md` - Main PRD (e.g., `002_phase2_feature_engineering_overview.md`)
- `{NNN}_phase{N}/{NNN}{letter}_{subtask}.md` - Task PRD (e.g., `002_phase2/002a_feature_extraction.md`)
- Completed PRDs move to `done/` folder

## Status Values

- **Draft** - In planning, not yet ready for implementation
- **In Progress** - Actively being implemented
- **Complete** - Implementation finished and validated
- **Superseded** - Replaced by newer version (moved to `done/`)

---

**Last Updated:** 2026-02-07
