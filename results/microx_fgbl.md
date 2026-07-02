# Microstructural Features for TLOB — FGBL (Bund) 1s & 5-tick

**Date:** 2026-07-01 · **Anchor instrument:** Bund (FGBL) · **Context:** Citi POC prep
(answers Tushar's "let's check if those help the DNN" — #citi-7july-call-preparation).

## What was added

New opt-in framework flag **`dataset.add_micro_extra`** (committed `e1f3e95` in
`~/PythonProjects/tlob`), appending **8 causal L1 features per source**
(`cst.N_MICRO_EXTRA_FEATURES`), composing with the existing `add_micro` block.
Ports this repo's Tushar linear-baseline signals (his "notional ratio / VWAP
difference / EWMA differences") into the DNN's input, plus a micro-price premium:

| Feature | Definition (Pa/Pb ask/bid px, Qa/Qb sizes, mid=(Pa+Pb)/2) | Origin |
|---|---|---|
| `notional_imb` | (Pb·Qb − Pa·Qa)/(Pb·Qb + Pa·Qa) ∈ [−1,1] | Tushar f1 (notional ratio) |
| `micro_prem` | micro-price − mid | micro-price premium |
| `sflow_ewma` | EWMA(side·qty), hl=5s | EWMA signed flow (Tushar f2) |
| `ofi_ewma` | EWMA(L1 OFI), hl=5s | EWMA OFI |
| `tobvwap_diff_f` / `_s` | tob_vwap − EWMA(tob_vwap), hl=5s / 30s | Tushar f3 (VWAP difference) |
| `mid_mom_f` / `_s` | mid − EWMA(mid), hl=5s / 30s | short-horizon momentum |

EWMAs are **time-aware** (decay by real trade Δt via `ewm(times=…)`), reset at each
UTC-day boundary; half-lives are tunable module constants
(`MICRO_EXTRA_HALFLIFE_FAST_S=5`, `SLOW=30` in `tlob/data/utils_data.py`). The count
is even so the total token dim stays even for TLOB's sinusoidal positional embedding.
Wired through `run.py` (feature count + `__microx` NPY cache suffix) and `dataloader.py`
mirroring `add_micro` exactly — existing consumers/caches untouched.

For FGBL single-source TBBO the input width is **22** = 4 raw L1 + 4 trade + 6
`add_micro` + 8 `add_micro_extra`.

## Targets

Both pure-sign (as a *maker* we don't cross the spread, so the spread+fee gate is
dropped): `dataset.label_threshold_ticks = 0.0` → `sign(Δ smoothed-mid)`, 3-class
up/flat/down (~balanced). Data: FGBL TBBO, 2025-06-02 → 2026-05-29 (~252 days),
chronological 80/5/15 split, OOS **raw argmax** (no confidence threshold).

- **1s**: `labeling_type=SMOOTHED_MID_TIME`, `horizon=1` (seconds), `len_trade_window=10`.
  Label dist [up, flat, down] = [0.286, 0.429, 0.285].
- **5-tick**: `labeling_type=SMOOTHED_MID`, `horizon=5` (events), `len_trade_window=10`.
  Label dist = [0.300, 0.402, 0.299].

## Results — features help both targets

| Target | Model | Test acc | Macro-F1 | Δ acc | Δ F1 |
|---|---|---|---|---|---|
| **1s sign** | baseline (raw L1) | 0.611 | 0.609 | — | — |
| **1s sign** | **+micro +microx (22 feat)** | **0.6210** | **0.6199** | **+1.0** | **+1.1** |
| **5-tick sign** | baseline (raw L1, 8 feat, `fgbl_smid_h5`) | 0.6814 | 0.6814 | — | — |
| **5-tick sign** | **+micro +microx (22 feat)** | **0.6889** | **0.6885** | **+0.75** | **+0.71** |

The `add_micro_extra` + `add_micro` block gives a **small but consistent ~+0.7–1.0 pt
lift** on both Bund targets — matching the "1–2%" order of magnitude Leo/Tushar expected.
The 5-tick sign target is far more predictable than 1s (0.69 vs 0.62 macro-F1): a
5-event smoothed-mid drift is more persistent than the 1-second one. OOS test sizes:
339,595 rows (1s), 752,354 rows (5-tick).

## Backtest (taker) — cost-dominated

Taker backtests crossing the ~0.80 bp round-trip spread + fees are deeply negative at
these horizons (1s microx Sharpe ≈ −74.5; 5-tick baseline taker ≈ −948, maker-reframed
`__mm` variant ≈ −34.6). This is a **maker** signal — the classification lift, not the
taker PnL, is the result here.

## Caveats

- The lift is `add_micro` **+** `add_micro_extra` combined vs raw L1. Isolating the new
  `add_micro_extra` block's marginal share needs an `add_micro`-only run (not yet done).
- Baselines: 1s from `results/mm_target_dnn_vs_linear.md` §2 (FGBL 0.611/0.609);
  5-tick from `checkpoints/backtest_overview.csv` variant `fgbl_smid_h5` (num_features=8),
  run 2026-06-30.

## Reproduction

```bash
# both targets, GPU per arg; pure-sign gate + full feature set
bash utils/run_fgbl_microx_one.sh 0 SMOOTHED_MID_TIME 1 fgbl_1s_microx     # 1s
bash utils/run_fgbl_microx_one.sh 1 SMOOTHED_MID      5 fgbl_5tick_microx  # 5-tick
# key overrides: dataset.add_micro=True dataset.add_micro_extra=True
#                dataset.label_threshold_ticks=0.0 experiment.fee=0.01
```

## Artifacts

- Feature block: `~/PythonProjects/tlob` commit `e1f3e95` (`add_micro_extra`).
- Run tags: `fgbl_1s_microx`, `fgbl_5tick_microx` → `checkpoints/TLOB/…__fgbl_*_microx/`.
- Logs: `data/logs/fgbl_1s_microx.log`, `data/logs/fgbl_5tick_microx.log`.
- Overview: `checkpoints/backtest_overview.csv`.
