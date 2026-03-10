# Urban Tree Transfer

Cross-city transfer learning for urban tree genus classification using
Sentinel-2 satellite imagery. Trains on Berlin, evaluates zero-shot on
Leipzig, then fine-tunes with small labeled samples to measure recovery.

Academic project — Geo Projektarbeit, 2025/26.

## Setup

```bash
uv sync
```

Requires Python 3.10+. GPU optional (CPU fallback available for CNN1D).

## Running

The pipeline runs in Google Colab via notebooks in `notebooks/runners/`.
Colab installs the package directly from the `main` branch — push all changes
before opening a notebook, otherwise Colab runs stale code.

Run in order: exploratory notebooks (`exp_XX_`) first, then runners
`01` → `02a/b/c` → `03a/b/c/d`. Each phase writes outputs to Google Drive;
download and commit to `outputs/` before starting the next phase.

## Development

```bash
uv run nox -s fix         # lint + format
uv run nox -s typecheck   # pyright
uv run nox -s test        # unit tests
uv run nox -s pre_commit  # run before every commit
```

See `docs/PROJECT.md` for research context and phase structure.
See `AGENTS.md` for conventions and project-specific rules.
