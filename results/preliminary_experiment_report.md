# TLOB on Eurex bond futures — overnight experiment campaign, 2026-06-11/12

## Setup (common to all experiments)

- **Data**: Databento XEUR.EOBI TBBO (top-of-book + trades), continuous front contracts,
  2025-06-09 → 2026-05-29 (the window shared by all six instruments). Train/val/test =
  80/5/15 chronological; test ≈ the last ~37 trading days (mid-Apr → end-May 2026).
- **Task**: 3-class direction of the smoothed mid over the next 60 seconds
  (up / flat / down), label threshold = 1 tick + fee. Session-gated 08:00–17:00 Berlin.
- **Model**: TLOB transformer, sequence of 256 book/trade events, ~4.3M params unless stated.
- **Evaluation protocol**: (1) per-tick ML metrics on the full test set at **argmax**
  (hundreds of thousands of ticks → statistically robust); (2) backtest only at a
  threshold giving **>30 round-trips** (τ=0.5 here) — backtests with ~30 trades are
  dominated by path noise and are NOT conclusive (we verified this directly, see #3/9/10).
  Backtest is net of exchange fee + half-spread per fill, no overnight carry.

**Metric legend** — *base rate*: fraction of test ticks whose label is up (resp. down);
*precision*: fraction of the model's up-signals that were truly up; *lift*: precision ÷
base rate (1.0 = no skill); *wrong-way*: fraction of up-signals where the truth was DOWN
(the expensive error); *WR/PF*: backtest win rate / profit factor (gross wins ÷ gross
losses, net of costs).

## What each experiment was

1. **fbtp_base** — Can TLOB predict the Italian BTP future from its own book alone?
   (BTP chosen as the most promising candidate: thinnest book of the liquid contracts,
   credit-spread volatility.)
2. **fbtp_stack6** — Same target, but the model sees all six books at once
   (Schatz, Bobl, Bund, Buxl, BTP, OAT) — the "information is in the other books" thesis.
3. **foat_base** — French OAT future from its own book alone. Run with **three different
   random seeds** to test whether results are training-noise-robust.
4. **foat_stack6** — OAT predicted from all six books.
5. **foat_lean3** — OAT predicted from only the books that *lead* it in our lead-lag
   analysis (Bund + BTP). Tests "input selection beats input quantity".
6. **fbtp_L8H1 / fbtp_L4H2** — Leo's scaling question: same 6-book FBTP setup with a
   2× larger model, grown two different ways (deeper: 8 layers; wider attention: 2 heads),
   both 8.7M params — a matched-capacity comparison.
7. **foat_h300** — OAT with a 5-minute horizon instead of 60s (is 60s just the wrong clock
   for a credit-driven instrument?).
8. **fgbx_base / fgbx_stack6** — Buxl (30y). Our lead-lag matrix says Buxl is the most
   *led* book in the complex, so if cross-book information helps anywhere it should be here.
9. **Lead-lag analysis** (no training) — cross-correlation of mid returns for all 30
   ordered pairs at 100ms and 1s grids over 60 days: who moves first, by how much.

## Results table

Per-tick ML metrics at argmax (full test set) + robust backtest (τ=0.5, >30 trades):

| # | run | up prec (lift) | up wrong-way | down prec (lift) | down wrong-way | trades | WR | PF |
|---|-----|----------------|--------------|------------------|----------------|--------|-----|-----|
| 1 | fbtp_base | 0.42 (2.4×) | 0.17 | 0.33 (1.9×) | 0.22 | 313 | 32% | 0.67 |
| 2 | fbtp_stack6 | 0.42 (2.3×) | 0.21 | 0.35 (2.0×) | 0.21 | 250 | 32% | 0.69 |
| 3 | foat_base seed1 | 0.39 (2.5×) | 0.14 | 0.28 (1.9×) | 0.28 | 210 | 31% | 0.70 |
| 3 | foat_base seed2 | 0.47 (3.0×) | 0.14 | 0.38 (2.6×) | 0.20 | 59 | 42% | 1.00 |
| 3 | foat_base seed3 | 0.44 (2.8×) | 0.17 | 0.27 (1.8×) | 0.36 | 159 | 35% | 0.81 |
| 4 | foat_stack6 | 0.44 (2.8×) | 0.17 | 0.33 (2.2×) | 0.22 | 305 | 26% | 0.63 |
| 5 | foat_lean3 | 0.40 (2.6×) | 0.15 | 0.29 (1.9×) | 0.20 | 276 | 28% | 0.57 |
| 6 | fbtp_L8H1 (8.7M deep) | 0.35 (2.0×) | 0.17 | 0.33 (1.9×) | 0.20 | 716 | 31% | 0.50 |
| 6 | fbtp_L4H2 (8.7M wide) | 0.39 (2.2×) | 0.17 | 0.33 (1.9×) | 0.20 | 319 | 32% | 0.68 |
| 7 | foat_h300 | 0.37 (1.2×) | 0.27 | 0.34 (1.2×) | 0.30 | 163 | 34% | 0.63 |
| 8 | fgbx_base | 0.39 (4.6×) | 0.08 | 0.31 (3.8×) | 0.16 | **91** | **52%** | **1.10** |
| 8 | fgbx_stack6 | 0.31 (3.5×) | 0.09 | 0.25 (2.7×) | 0.18 | 249 | 33% | 0.74 |

Lead-lag (#9): **Bund leads the entire complex** (strongest net-lead at both grids), BTP
second; everything propagates within ~1 grid step (sub-second). BTP leads OAT more than
Bund does. Peak cross-correlations are small (~0.02–0.05): real structure, but weak.

## Conclusions

1. **The model learns real per-tick signal everywhere at h=60s**: up-side precision is
   2–4.6× the base rate, reproducible across seeds (seeds 1/2/3 of foat_base: 2.5×/3.0×/2.8×).
   The 5-minute horizon (#7) is the exception — lift collapses to 1.2× → 60s is the right clock.
2. **The signal does not survive the strategy layer**: at robust trade counts (>30, up to
   716 round-trips) every config except one is PF < 1. Costs (spread paid both legs) plus
   the hold-until-opposite-signal exit eat a 2–3× precision edge whose wrong-way rate is
   ~0.15–0.2. Important methodological note: with only ~30 trades, backtests swing from
   PF 3.4 to PF 0.4 on the same underlying signal quality — we now evaluate on per-tick
   metrics first, and only trust backtests above 30 trades.
3. **More inputs did not help**: 6-book and leaders-only stacks match the single-book
   models on per-tick precision and do worse in the backtest, on both BTP and OAT.
   Same for model size: 2× capacity (deeper or wider) slightly *lowers* precision —
   the 4.3M model is not capacity-limited on ~2M training samples.
4. **One open candidate: Buxl single-book (#8)** — highest precision lift of the campaign
   (4.6×/3.8×), lowest wrong-way rates (0.08/0.16), and the only robust-count backtest
   above water (91 trades, 52% WR, PF 1.10). Single seed, modest PF, wide-tick instrument —
   needs the seed-robustness treatment before it counts. The irony: lead-lag says Buxl is
   the most *led* book, and it's the most predictable from its own book alone.
5. **Where this points next**: the per-tick edge is real but the outright-trading
   formulation wastes it. Candidates, in our order of preference: (a) spread/pair
   prediction and trading (Bund–Bobl, 30y–5y — Tushar's portfolio idea), (b) entry/exit
   logic that doesn't pay the spread twice per round-trip blindly, (c) Tushar's tapered
   "N updates before the move" labeling, (d) Buxl seed-robustness + a closer look.
