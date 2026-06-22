# tlob-eurex

LOB directional prediction for **Eurex euro government-bond futures** — German Bund/Bobl/Schatz/Buxl plus Italian BTP and French OAT.

This is a thin **consumer repo** of the general [`tlob`](../tlob) framework (sibling checkout, editable-installed). It owns nothing but the Eurex instrument economics and a handful of Eurex-specific analyses; all training, preprocessing, models, and backtesting live in `tlob`. It mirrors the structure of the A2A commodity consumer [`tlob-a2a`](../tlob-a2a) — see that repo's `CLAUDE.md` for the shared framework conventions.

## What it does

Trains the **TLOB transformer** to predict the 3-class direction (up / flat / down) of the smoothed mid over a fixed horizon, from a sequence of top-of-book + trade events. Models can be trained on a single instrument's book or on **stacked multi-instrument books** (the "information is in the other books" thesis), and evaluated both with per-tick ML metrics and cost-aware backtests.

## Instruments

`instruments_eurex.py` registers six continuous-front futures into `tlob.instruments` (the library ships zero entries). All are `€1000`/point, EUR, session 08:00–17:00 Europe/Berlin, Databento `XEUR.EOBI`:

| Root | Instrument | Tick |
|------|-----------|------|
| FGBL | Bund 10y (DE)   | 0.01 |
| FGBM | Bobl 5y (DE)    | 0.01 |
| FGBS | Schatz 2y (DE)  | 0.005 |
| FGBX | Buxl 30y (DE)   | 0.02 |
| FBTP | BTP 10y (IT)    | 0.01 |
| FOAT | OAT 10y (FR)    | 0.01 |

`fee=0.01` = 1 tick for FGBL/FGBM/FBTP/FOAT; Schatz tick is 0.005, Buxl 0.02 — set `experiment.fee` accordingly.

## Layout

- `main.py` — thin Hydra entry: imports `instruments_eurex` for side-effects (populates the registry), then delegates to `tlob.run` with tlob's base `Config`. No custom config extensions.
- `instruments_eurex.py` — the six registry entries (economics + session).
- `sweep.py` — τ-sweep wrapper around `tlob.backtest.rerun_backtest`; registers the Eurex instruments first so fees/sessions resolve.
- `analysis/` — standalone analyses:
  - `lead_lag.py` — cross-instrument order-book lead-lag matrix (who moves first).
  - `ml_metrics.py` — per-tick precision/recall/F1 on the full test set, gated like the strategy.
- `results/` — analysis outputs and the campaign report (`experiment_report.md`).
- `data/{SYM}_TBBO/` — per-day parquet **symlink farms** into `~/PythonProjects/tlob-a2a/data/XEUR.EOBI/{SYM}.v.0/`. FBTP/FOAT are clamped to ≤2026-05-29 so all six sources share the same window (German TBBO ends there).
- `data/NPY/`, `checkpoints/`, `data/logs/` — preprocessed shards, checkpoints, logs (gitignored, same roles as in tlob-a2a).

## Environment

No own venv yet — run with the shared interpreter (has `tlob` editable + all deps):

```
/home/rig/PythonProjects/tlob-a2a/.venv/bin/python
```

## Canonical run protocol

Single-source (predict FBTP from its own book):

```bash
/home/rig/PythonProjects/tlob-a2a/.venv/bin/python main.py +model=tlob \
  experiment.fee=0.01 dataset.is_databento=True experiment.filter_session=True \
  dataset.batch_size=1024 experiment.is_wandb=False experiment.max_epochs=20 \
  dataset.data_sources=[FBTP_TBBO] dataset.n_lob_levels=1 \
  dataset.labeling_type=SMOOTHED_MID_TIME experiment.horizon=60 dataset.len_trade_window=10 \
  experiment.is_data_preprocessed=False experiment.run_tag=fbtp_base hydra.job.chdir=False
```

Stacked multi-source — **anchor (predicted) instrument first**, `experiment.is_3d=False`; the builder intersects common hours across sources:

```bash
  dataset.data_sources=[FBTP_TBBO,FGBL_TBBO,FGBM_TBBO,FGBS_TBBO,FGBX_TBBO,FOAT_TBBO]
```

Pick the free GPU with `experiment.gpu_id=<n>`.

## Gotchas

- **TBBO is top-of-book only** → `n_lob_levels=1` always.
- `filter_session=True` adds `__sess` to the NPY cache key.
- German TBBO ends **2026-05-29**; FBTP/FOAT raw extends to 2026-06-08 but the farms here are clamped — rebuild the farms for the extra week on single-source runs.
- GPU0 (RTX 3090, bottom slot) has fallen off the bus under dual-GPU load — prefer GPU1 for the longest unattended runs.

## Results

See `results/experiment_report.md` for the overnight campaign (2026-06-11/12): single-book vs. stacked-book, lead-lag-driven input selection, model scaling, and horizon sweeps across the six instruments. Evaluation judges models on **per-tick test-set ML metrics** (statistically robust over hundreds of thousands of ticks), with backtests used only above a >30-round-trip threshold (fewer trades are dominated by path noise).
