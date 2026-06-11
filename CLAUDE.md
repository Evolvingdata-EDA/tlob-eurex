# tlob-eurex — Eurex government-bond futures consumer repo

LOB directional prediction for Eurex euro govvie futures (Bund/Bobl/Schatz/Buxl + BTP/OAT).
Consumer of the general `tlob` framework (sibling checkout `~/PythonProjects/tlob`,
editable-installed). Mirrors the structure of `~/PythonProjects/tlob-a2a` (the A2A
commodity consumer) — read that repo's CLAUDE.md for framework conventions; they all apply.

## Environment
**No own venv yet** — run with the shared interpreter:
`/home/rig/PythonProjects/tlob-a2a/.venv/bin/python` (has `tlob` editable + all deps).
Make standalone later if this repo grows.

## Layout
- [main.py](main.py) — thin Hydra entry; imports `instruments_eurex` for side-effects,
  delegates to `tlob.run` with tlob's base `Config` (no custom config extensions).
- [instruments_eurex.py](instruments_eurex.py) — registry entries for FGBL/FGBM/FGBS/FGBX/FBTP/FOAT
  (€1000/point, €0.40/side placeholder, session 08–17 Europe/Berlin).
- `data/{SYM}_TBBO/` — per-day parquet **symlink farms** into
  `~/PythonProjects/tlob-a2a/data/XEUR.EOBI/{SYM}.v.0/`. FBTP/FOAT are clamped to
  ≤2026-05-29 so all six sources share the same window (German TBBO ends there).
- `analysis/` — standalone analyses (lead-lag matrix etc.). `results/` — outputs.
- `data/NPY/`, `checkpoints/`, `data/logs/` — same roles as in tlob-a2a.

## Canonical run protocol (mirrors the FGBL 06-09 experiments)
```bash
/home/rig/PythonProjects/tlob-a2a/.venv/bin/python main.py +model=tlob \
  experiment.fee=0.01 dataset.is_databento=True experiment.filter_session=True \
  dataset.batch_size=1024 experiment.is_wandb=False experiment.max_epochs=20 \
  dataset.data_sources=[FBTP_TBBO] dataset.n_lob_levels=1 \
  dataset.labeling_type=SMOOTHED_MID_TIME experiment.horizon=60 dataset.len_trade_window=10 \
  experiment.is_data_preprocessed=False experiment.run_tag=<tag> hydra.job.chdir=False
```
Stacked: `dataset.data_sources=[FBTP_TBBO,FGBL_TBBO,...]` — **anchor (predicted) first**,
`experiment.is_3d=False`. The multi-source builder intersects common hours across sources.

## Gotchas inherited from the Eurex work in tlob-a2a
- TBBO is top-of-book only → `n_lob_levels=1` always.
- fee=0.01 = 1 tick (FGBL/FGBM/FBTP/FOAT); Schatz tick is 0.005, Buxl 0.02.
- `filter_session=True` adds `__sess` to the NPY cache key.
- German TBBO ends 2026-05-29; FBTP/FOAT raw extends to 2026-06-08 but the farms here
  are clamped — rebuild the farms if you want the extra week for single-source runs.
- GPU0 (RTX 3090, bottom slot) fell off the bus twice on 2026-06-09 under dual-GPU load;
  fine since reboot, but prefer GPU1 for the longest unattended runs.
