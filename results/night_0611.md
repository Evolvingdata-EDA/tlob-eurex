# Overnight experiments 2026-06-11 → 12 — BTP/OAT + TLOB scaling

Protocol: TBBO L1, SMOOTHED_MID_TIME h=60s, lw=10, fee=0.01, session 08–17 Berlin,
STANDARD 80/5/15 split, window 2025-06-09 → 2026-05-29 (all sources clamped to the
common German end-date). Test ≈ last ~37 trading days. Backtest net of fees + half-spread,
flatten-at-close, no overnight carry. Engine default τ = 0.6.

Labels (FBTP anchor): 14.8% up / 70.2% hold / 15.1% down — ~30% directional vs ~19%
for FGBL at the same h60 (periphery moves more per unit time).

## Run results

### stack6→FBTP, TLOB L4H1 4.36M (control) — `fbtp_stack6_h60_sess`
val 0.904 (epoch 1, early-stopped), test_loss 0.861

| τ | trades | WR | Net PnL € | Sharpe | PF | bps/trade |
|---|---|---|---|---|---|---|
| 0.60 | 24 | 58.3% | +135.70 | 0.41 | 1.07 | +0.06 |
| 0.65 | 11 | 54.5% | −59.80 | −0.27 | 0.93 | −0.05 |
| 0.70 | 1 | 100% | +125.77 | — | — | — |
| 0.75 | 0 | — | — | — | — | — |

Verdict: thin/flat — better than German stacked-4 (which went negative as trades rose)
but 24 trades at PF 1.07 is not an edge yet.

### FBTP→FBTP baseline L4H1 4.28M — `fbtp_h60_sess`
val 0.910 (epoch 2, early-stopped), test_loss 0.864

| τ | trades | WR | Net PnL € | Sharpe | PF | bps/trade |
|---|---|---|---|---|---|---|
| 0.60 | 31 | 45.2% | +273.44 | 0.56 | 1.13 | +0.09 |
| 0.65 | 7 | 57.1% | −175.02 | −1.22 | 0.69 | −0.25 |
| 0.70 | 0 | — | — | — | — | — |
| 0.75 | 0 | — | — | — | — | — |

Verdict: same thin-positive zone as the stacked control; probabilities compress fast
(dead above τ0.6). Single-source ≈ stacked-small at matched trade count so far —
the multi-source gain hasn't shown up at 4M params.

### FOAT→FOAT baseline L4H1 4.28M — `foat_h60_sess`  ★ best of the night
val 0.898 (epoch 3), test_loss 0.797. Labels 13.5/72.8/13.7.

| τ | trades | WR | Net PnL € | Sharpe | PF | bps/trade |
|---|---|---|---|---|---|---|
| 0.60 | 31 | 41.9% | +914.73 | 1.43 | 1.35 | +0.30 |
| 0.65 | 12 | 41.7% | +738.27 | 3.58 | 3.42 | +0.62 |
| 0.70 | 6 | 33.3% | +64.68 | 0.90 | 1.45 | +0.11 |
| 0.75 | 0 | — | — | — | — | — |

Verdict: **the healthy τ-monotone shape** (positive at every τ, PF rises with confidence) —
the same signature as the good TFM configs. OAT ≫ BTP for single-source TLOB. Consistent
with lead-lag: FOAT is the most-led periphery book (followers are predictable). Trade
counts still small — needs more test days / live validation before any deployment talk.

**⚠ Seed-robustness check (seed2, same config/NPY):** val 0.908 / test 0.799 (consistent
losses), but the trading behavior flips — only 4 trades at τ0.6 (+678, PF 2.18, noise-n),
and at the matched-count diagnostic τ0.55 (25 trades): **−1,224 EUR, PF 0.67, 36% WR**.
At similar trade counts seed1 was +915/PF 1.35. The FOAT edge does NOT survive seed2;
seed3 running as arbiter. Treat the seed1 ladder as optimistic until proven otherwise.

**Seed3 (arbiter): NEGATIVE.** val 0.895 / test 0.795 (best raw losses of the three!) but
τ0.6: 23 trades, −1,315.99 EUR, Sharpe −3.26, PF 0.42; τ0.65: 4 trades, −907, PF 0.01.

**FINAL FOAT h60 verdict: 1-for-3 across seeds → the seed1 result was luck, not edge.**
Notable: the seed with the BEST val/test losses (seed3) trades worst — CE loss does not
rank PnL on this data at all. Combined with the German results, the Eurex govvie complex
has no robust single-book or stacked-book TLOB edge at h=60s, at any capacity tried.

### stack6→FOAT L4H1 — `foat_stack6_h60_sess`
val 0.882 (ep1) ≪ baseline 0.898; test_loss 0.791 < 0.797 — but PnL is WORSE:

| τ | trades | WR | Net PnL € | Sharpe | PF |
|---|---|---|---|---|---|
| 0.60 | 32 | 34.4% | +353.27 | 0.97 | 1.16 |
| 0.65 | 8 | 37.5% | −71.81 | −0.58 | 0.85 |
| 0.70 | 0 | — | — | — | — |

Verdict: **full 6-stack hurts FOAT** at every matched trade count (PF 1.16 vs 1.35 @τ0.6;
0.85 vs 3.42 @τ0.65) and probabilities compress (dead at τ0.7 where baseline still traded).
Better CE loss, worse PnL — the German wings add classification signal on easy flats but
noise on the directional tails. Input selection > input quantity.

### lean3 [FOAT+FGBL+FBTP]→FOAT — `foat_lean3_h60_sess`
val 0.885, test_loss 0.797 (= baseline).

| τ | trades | WR | Net PnL € | Sharpe | PF |
|---|---|---|---|---|---|
| 0.60 | 35 | 28.6% | −345.12 | −0.48 | 0.88 |
| 0.65 | 12 | 33.3% | −291.59 | −1.06 | 0.73 |
| 0.70 | 3 | 100% | +610.33 | 3.97 | — |

Verdict: lean3 also loses to the baseline — at 12 trades it's PF 0.73 vs the baseline's
PF 3.42 at the same count. **Multi-source is 0-for-3 on periphery anchors tonight**
(6-stack→FBTP ≈ flat, 6-stack→FOAT worse, lean3→FOAT worse). The single OAT book
beats every input combination tried. Single seed, ~37 test days — treat as a strong
hint, not a theorem; seed/window robustness checks are the obvious daylight follow-up.

### stack6→FBTP, L8H1 8.7M depth-scaled — `fbtp_stack6_h60_L8H1_sess`
val 0.909 (ep1) vs control 0.904; test_loss 0.870 vs 0.861.

| τ | trades | WR | Net PnL € | Sharpe | PF |
|---|---|---|---|---|---|
| 0.60 | 79 | 35.4% | −111.81 | 0.06 | 0.98 |
| 0.65 | 23 | 39.1% | −104.41 | −0.24 | 0.95 |
| 0.70 | 5 | 40.0% | +456.19 | 1.76 | 1.90 |

Verdict: **depth alone hurts** — at matched trade count (23 vs control's 24) PF 0.95 vs 1.07,
and val/test losses both worse. τ0.7 positive is 5 trades = noise.

### stack6→FBTP, L4H2 8.7M heads-scaled — `fbtp_stack6_h60_L4H2_sess`
val 0.909 (ep1) — identical to L8H1's best; test_loss 0.871. τ0.6: 24 trades, 45.8% WR,
−155.55 EUR, Sharpe −0.09, PF 0.91.

**Scaling verdict (Leo's request)**: at matched capacity (8.7M), heads ≈ depth — val
0.909 both — and both lose to the 4.3M control (val 0.904, PF 1.07 vs 0.91/0.95 at
matched trade count). TLOB is NOT capacity-limited on this data (~2.2M train samples);
L8H2 (17.4M) dropped as pointless. If width scaling is still wanted, it needs the
`hidden_dim` framework patch (model width is hard-locked to num_features today) —
prepared as a daylight proposal, not run overnight (live paper-trading shares the
tlob editable install via the 07:50 restart).

### FOAT h300 baseline — `foat_h300_sess`
Labels 27.5/45.0/27.5 (55% directional). val 1.107, test 1.081 (chance for this mix ≈ 1.099
unweighted — barely under). τ0.6: 3 trades, −268 EUR, PF 0.08. **h300 dead on the periphery
too** — mirrors German h300. The horizon axis is closed: 60s is the right clock and it's thin.

### FGBX (Buxl) baseline + stacked6 — `fgbx_h60_sess` / `fgbx_stack6_h60_sess` (fee=0.02)
Baseline: val 0.674, test 0.547; τ0.6: 15 trades, −2,147 EUR, Sharpe −3.22, PF 0.41.
Stacked6: val 0.684, test 0.593 (both worse); τ0.6: 51 trades, −3,654 EUR, Sharpe −5.55, PF 0.51.

Verdict: Buxl is the worst instrument of the night, and the **lead-lag thesis test failed** —
even on the most-led book, cross-book stacking made things worse, not better. Lead-lag
correlations (~0.02-0.05) are real but too small to clear a 2-tick-cost instrument.

## Night conclusions (12 trainings, all swept)

1. **No robust TLOB edge anywhere on the Eurex govvie complex at h60 or h300** — German
   core (prior work) AND periphery (tonight), single-source and multi-source, 4.3M→17.4M
   params. The one positive (FOAT seed1, PF 1.35/3.42) failed seed robustness 1-for-3.
2. **Scaling refuted** (Leo's request): depth (L8) and heads (H2) at matched 8.7M capacity
   both lose to the 4.3M control on val, test, and PnL. The model is not capacity-limited;
   the `hidden_dim` width patch is NOT worth doing on this evidence.
3. **Multi-source stacking refuted on this complex**: 0-for-5 (FBTP 6-stack, FOAT 6-stack,
   FOAT lean3, FGBX 6-stack; German 4-stack prior). Better CE loss ≠ better PnL, repeatedly.
4. **CE-loss/PnL decoupling**: the best-loss seed traded worst. Checkpoint selection by
   val_loss is unreliable for trade quality here.
5. **Lead-lag structure exists** (FGBL leads all, 1-step propagation) but its corr magnitude
   (≲0.05) doesn't survive costs as a directional signal at these horizons.

**What's left untried (daylight, needs framework work / decisions):** spread/pair
prediction (Tushar's portfolio idea — label on Bund-Bobl or 30y-5y spread, trade DV01-ish
pairs); Tushar's tapered 10-updates-before-move labeling; MBP-10 depth features (only
~21 days of data); event/news-conditioned models. These are different *problems*, not
re-runs — the plain "predict next move from the book(s)" formulation is exhausted here.

## Lead-lag (done, 60d: 2026-03-04 → 05-29) — `results/lead_lag/`
FGBL leads the complex (net +0.055 @1s), FBTP second (+0.037), all best lags = 1 step.
Most-led: FGBX/FGBS/FOAT → FGBX queued as multi-source target. FBTP→FOAT stronger than
FGBL→FOAT at 100ms (periphery internal pecking order).
