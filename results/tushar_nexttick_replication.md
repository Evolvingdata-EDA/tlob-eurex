# Replicating Tushar's Next-Tick Label — TLOB vs his Linear Model (Bund)

**Date:** 2026-06-30 · **Instrument:** Bund (FGBL) · **Context:** Citi POC prep.

## Question

Tushar's [EurexNextTickSignalModeling](../../EurexNextTickSignalModeling) predicts the
**sign of the next settled one-tick mid move** with a transparent linear model
(~92% on the modelled set ≤200 events ahead; ~76% on the full no-horizon-cap set).
Can **TLOB beat his linear model on his own labelling method**, on the Eurex data we
hold? This is the *opposite* regime to the earlier
[mm_target work](mm_target_dnn_vs_linear.md): that used a 1s-smoothed time target where
TLOB won by +12pts; here the horizon is the **next tick**.

## What "his labelling method" is (replicated exactly)

From his [grid.py](../../EurexNextTickSignalModeling/src/nt_signals/grid.py):

- **Event** = a top-of-book change. **Target** = the sign of the next *settled*
  (1-tick-spread) mid that differs from the current mid — searched strictly forward
  within the trading day. Non-zero by construction → **binary up/down, no flat class**.
- Only **1-tick-spread** events are labelled (the modal "settled" state); wide-spread
  transients are skipped.
- **No d_max horizon cap** (per Leo: 200 ticks ≈ the median d, and the ~92%→76% drop
  from capping at 200 implies the >200 bucket sits well below 76%; so 76% is the honest
  blended bar). Every settled event with a later same-day move is labelled.
- Scored by **sign accuracy**.

### Framework implementation (`~/PythonProjects/tlob`)

New `LabelingType.NEXT_TICK` in the shared framework (mirrors the `label_threshold_ticks`
precedent):
- `utils_data.labeling()` — vectorised per-day settled-move logic; emits `0`=up / `2`=down,
  `NaN` (→ dropped) for non-settled events and end-of-day no-move. 3-class container with
  the flat class never populated, so the backtest/metrics paths and the prior mm TLOB
  numbers stay directly comparable; balanced sampling skips the empty class.
- `get_label_lookback` returns **1** for NEXT_TICK → the label anchor coincides with the
  input window's last row, predicting the next move strictly *after* it (leakage-free).
- 5 unit tests (`tlob/tests/test_next_tick_labeling.py`) verify the label against an
  independent brute-force reference of his grid.py semantics, day-boundary handling, the
  1-tick-spread filter, and that only up/down are emitted.

**Label sanity:** 4,997,920 labelled Bund events over 252 days; distribution **50.16% up /
49.84% down** (flat 0%) — balanced binary, as expected with no cap.

## 1. Head-to-head (252-day TBBO L1, chronological 80/5/15, same ~749k test rows)

| Model | Input / features | Sign accuracy |
|---|---|---|
| Linear — imbalance only (f1) | 1 feature | 0.7924 |
| Linear — f1 + f3 (imbalance + momentum) | 26 features | 0.7923 |
| Linear — all 40 (his L1 set) | 40 features | 0.7926 |
| **TLOB (deep)** | raw L1 LOB sequence | **0.7932** |
| Tushar reported (his MBP-10 June data) | his 90-feature OLS | ~0.76 full / ~0.92 ≤200d |

Linear = multinomial/binary logistic on his features built with his exact `ewma_vec`
(f1 imbalance, f2 trade intensity, f3 quote-momentum ×5 half-lives, f5 lead-lag, f6
move-from-open; f4 depth excluded — needs L10 books, unavailable on TBBO).

**TLOB ties the linear model (~79.3% both) — it does *not* win here.** And the linear
signal **saturates at a single feature**: imbalance alone (79.24%) ≈ all 40 (79.26%).
This is the sharpest possible confirmation of Tushar's "near-mechanical at short
horizons" caveat — the next-tick sign is almost fully determined by the *current*
top-of-book imbalance, a static feature both models capture. A sequence model has
essentially nothing to add at this horizon (vs +12pts at the 1s horizon, where the
mechanical signal has decayed).

> Note both clear his ~76%: that is *not* TLOB beating him — it reflects a different
> data window (our 252-day TBBO L1, 2025-06→2026-05) and the no-cap full set on our tape.
> The honest comparison is TLOB vs linear on the **same** rows: a statistical tie.

## 2. Does order-book depth help at the next-tick horizon? (MBP-10, ~40 days)

MBP-10 (10-level depth) downloaded for Bund 2026-04-30→2026-06-26 (~40 trading days,
incl. the June window Tushar actually used). One TLOB trained on full L10 depth
(44 features), NEXT_TICK label, same chronological split. (Per Leo: no separate matched
L1 control — compared against the L1/linear numbers above. Caveat: ~40-day window vs the
252-day L1 run, so this is depth *and* far less data, not a clean depth isolation.)

| TLOB input | Clock | Levels | # feats | Train rows | Sign accuracy (test) |
|---|---|---|---|---|---|
| TBBO | trade | L1 | 8 | 4.0M (252d) | **0.7932** |
| MBP-10 | tob_priceqty | L10 (full depth) | 44 | 40.1M (~40d) | **0.7472** |

MBP-10 is sampled on `mbp_clock="tob_priceqty"` (every TOB change = Tushar's event
definition; the trade clock keeps only ~2.5% of book events). 40.1M train rows (dense
clock over ~40 days = ~10× the 252-day TBBO trade-clock set), seq 512, lean float32 build
(`dataset.lean_float32=True`, added to the framework to fit the ~50M-row build in RAM).

**Depth hurts (−4.6 pts) and overfits fast.** Val accuracy peaked at 75.3% after only 10k
steps (26% of epoch 0), then collapsed as train loss kept falling (val_loss 0.52 → 1.33 by
epoch 2 → patience-8 early stop). Best-checkpoint test = 0.7472, well below the L1/linear
~0.79. Caveat: the L10 run uses the dense clock while the L1/linear numbers use the trade
clock, so this is depth **+** clock, not a pure depth isolation (the matched L1-dense
control was deferred) — but the overfitting signature is decisive on its own.

## Takeaways

1. The next-tick label is **near-mechanical**: imbalance alone ≈ 79% on our data; the other
   39 L1 features add +0.02pts.
2. **TLOB ≈ linear (~79.3%)** at the next-tick horizon — no DNN edge, the opposite of the
   1s-smoothed target (where TLOB won by +12pts). A sequence model only adds value as the
   horizon lengthens and the static imbalance signal decays.
3. **Order-book depth (MBP-10 L10) does not help — it hurts (0.7472 vs 0.7932) and overfits.**
   Consistent with the earlier mm-target depth finding. Top-of-book L1 is enough (and is the
   only feed available across all instruments and the full year).

## Artifacts

- `analysis/tushar_nexttick_baseline.py` — linear baseline (his features + `ewma_vec`) on
  the NEXT_TICK label.
- `analysis/download_june_fgbl.py` — June FGBL mbp-10/tbbo pull (databento, $0).
- `results/tushar_nexttick_features.parquet` — cached 40-feature matrix.
- Framework: `LabelingType.NEXT_TICK` in `~/PythonProjects/tlob` (+ 5 unit tests).
- Tushar's repo: `~/PythonProjects/EurexNextTickSignalModeling`.
