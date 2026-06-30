# Market-Making Target — TLOB vs Tushar's Linear Model (Eurex bond futures)

**Date:** 2026-06-26 · **Anchor instrument:** Bund (FGBL) · **Context:** Citi POC prep.

## Target

Market-making reframing of the next-tick signal. Tushar's label gates the move on
`spread + fee`, which at short horizons is ~97% "flat" (almost nothing clears ~2 ticks
in a second) → degenerate to learn. As a *maker* we don't cross the spread, so the gate
is dropped and the label is the **direction of the next-second smoothed-mid drift**:

- `labeling_type = SMOOTHED_MID_TIME`, `horizon = 1s`, `len_trade_window = 10`
- new framework param `dataset.label_threshold_ticks = 0.0` → pure `sign(Δ smoothed-mid)`
  (absolute tick gate, replaces `spread + fee`; `fee` freed for the backtest)
- 3-class: up / flat / down (~balanced)

All metrics below are **out-of-sample**, raw **argmax** (no confidence threshold),
chronological 80/5/15 split.

## 1. TLOB vs linear baseline (head-to-head)

Identical Bund target, split, and OOS test rows (~340k). Linear = multinomial logistic
on Tushar's features built with his exact `ewma_vec` (f1 imbalance, f2 trade intensity,
f3 quote-momentum ×5 half-lives, f5 lead-lag, f6 move-from-open = **40 features**; depth
family f4 excluded — needs L10 books unavailable over the full window).

| Model | Input / features | Accuracy | Macro-F1 | up F1 | flat F1 | down F1 |
|---|---|---|---|---|---|---|
| **TLOB (deep)** | raw L1 LOB sequence | **0.611** | **0.609** | 0.596 | 0.669 | 0.561 |
| **Linear (logistic)** | all 40 (his full L1 set) | 0.489 | 0.472 | 0.554 | 0.316 | 0.547 |
| **Δ (TLOB − linear)** | | **+0.122** | **+0.137** | +0.042 | +0.353 | +0.014 |

**TLOB beats the linear model by +12.2 pts accuracy / +13.7 pts macro-F1.** The largest
gap is the **flat** class (+0.35 F1): the linear model can barely detect "no-drift"
periods (0.316) while TLOB reads them well (0.669); on pure direction the gap is smaller
but still favors TLOB.

### Linear feature ablation (same target/split)

| Linear features | Accuracy | Macro-F1 | up F1 | flat F1 | down F1 |
|---|---|---|---|---|---|
| f1 only (imbalance) | 0.441 | 0.396 | 0.506 | 0.177 | 0.506 |
| f1+f3 (imbalance + momentum) | 0.492 | 0.468 | 0.561 | 0.291 | 0.551 |
| all 40 (his full L1 set) | 0.489 | 0.472 | 0.554 | 0.316 | 0.547 |

The linear signal **saturates at f1+f3** (both top-of-book). Imbalance — which scores ~92%
on Tushar's near-mechanical *next-tick* target — reaches only ~49% (3-class) on the
forward-looking *1s smoothed* target. TLOB's sequence model captures the rest.

## 2. Cross-instrument TLOB (1s sign target, 252-day TBBO)

| Instrument | Tenor | Accuracy | Macro-F1 | up F1 | flat F1 | down F1 |
|---|---|---|---|---|---|---|
| Bund (FGBL) | 10y DE | 0.611 | 0.609 | 0.596 | 0.669 | 0.561 |
| Bobl (FGBM) | 5y DE | 0.580 | 0.576 | 0.566 | 0.634 | 0.529 |
| BTP (FBTP) | 10y IT | 0.571 | 0.521 | 0.602 | 0.332 | 0.631 |
| Buxl (FGBX) | 30y DE | 0.561 | 0.546 | 0.586 | 0.435 | 0.616 |
| OAT (FOAT) | 10y FR | 0.558 | 0.541 | 0.592 | 0.437 | 0.594 |
| Schatz (FGBS) | 2y DE | 0.543 | 0.542 | 0.513 | 0.578 | 0.536 |

Same **liquidity gradient** as Tushar's report (German core on top, Schatz weakest).
BTP/Buxl/OAT have low flat-F1 but the sharpest directional F1 — their mid rarely sits
still at 1s, so accuracy is dragged by a thin flat class.

## 3. Does order-book depth help? (matched 21-day window)

MBP-10 (L10 depth) exists only for Bund, only 21 days (2026-04-30→05-29). To isolate
depth from data size, both models trained on the **same 21 days, same target/split**.

| TLOB input | Levels | # feats | Accuracy | Macro-F1 | up F1 | flat F1 | down F1 |
|---|---|---|---|---|---|---|---|
| TBBO | L1 (top of book) | 8 | **0.557** | **0.555** | 0.565 | 0.606 | 0.493 |
| MBP-10 | L10 (full depth) | 44 | 0.497 | 0.495 | 0.461 | 0.534 | 0.491 |

**Depth does not help — it hurts (−6.0 pts).** More inputs (44 vs 8) on far less data
(21 days) is less data-efficient, and depth carries little marginal signal at 1s
(corroborated by the linear ablation saturating at L1). **Use TBBO / L1** — also the only
feed available across all instruments and the full year. (Caveat: only 1 month of MBP-10
exists; a full year of depth could shift this, but untestable now.)

## Takeaways

1. The MM reframing makes the 1s direction target learnable (vs 97%-flat under spread+fee).
2. TLOB extracts genuine forward-looking 1s direction across the curve.
3. **TLOB beats Tushar's linear approach by +12 pts** on the identical target — the DNN case.
4. Depth (MBP-10) is not worth it at 1s; top-of-book L1 is enough.

## Artifacts

- `analysis/label_dist_mm.py` — label-distribution sweep (sign / ½-tick / 1-tick).
- `analysis/tushar_linear_baseline.py` — linear baseline (his features + `ewma_vec`).
- `results/tushar_linear_features.parquet` — cached 40-feature matrix.
- `checkpoints/backtest_overview.csv` — all runs (variants `mm_h1_sign*`).
- Framework change: `dataset.label_threshold_ticks` in `~/PythonProjects/tlob`.
- Tushar's repo: `~/PythonProjects/EurexNextTickSignalModeling`.
