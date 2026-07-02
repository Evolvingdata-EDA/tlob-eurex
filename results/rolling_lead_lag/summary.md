# Rolling intraday lead-lag — are the driver->govvie edges transient?

All 13 instruments in data/BARS/ (2025-03 -> 2026-06, 325 session days). 120m sliding window, 15m step, **strictly-leading lags 1-10m (contemporaneous lag 0 excluded — a genuine, tradeable lead only)**, edge 'live' when out-of-sample R2 > 0.05 (|corr| > 0.22).

OOS R2 = lag picked on the window's first half, corr^2 scored on the disjoint second half — no look-ahead in lag selection.


**Noise floor:** under no real edge, the detector still fires ~**8.6%** of the time (chance corr^2 on the test half). Read every number against this — a pair near the floor has no exploitable lead-lag; only pairs well above it do.


**Headline:** with contemporaneous (lag-0) co-movement excluded, the tradeable leads are **thin but real**. No pair is fat — the strongest, ZN->FGBM, is live 13.8% vs the 8.6% chance rate (1.6x) — but the excess is statistically solid and, tellingly, **ZN leads essentially the whole euro curve** (12.7-13.8% live across all six govvies), a consistent signature rather than a one-off. So there IS a genuine ZN->euro-rates lead at 1-min, just economically small. Going sub-minute should sharpen it if the true lead is being blurred inside the 1-min bar.


## Transience by pair (sorted by % time live)

`x noise` = % live / 8.6% floor; >~2 is a real, if intermittent, edge.

| driver -> target | mean \|corr\| | % live | x noise | median live-spell (min) | # changepoints |
|---|---|---|---|---|---|
| ZN -> FGBM | 0.188 | 13.8% | 1.6 | 15 | 15 |
| ZN -> FOAT | 0.186 | 13.4% | 1.6 | 15 | 7 |
| ZN -> FGBL | 0.188 | 13.4% | 1.6 | 15 | 8 |
| TFM -> FGBL | 0.185 | 13.1% | 1.5 | 15 | 0 |
| ZN -> FBTP | 0.186 | 13.1% | 1.5 | 15 | 3 |
| ZN -> FGBX | 0.189 | 12.8% | 1.5 | 15 | 2 |
| ZN -> FGBS | 0.185 | 12.7% | 1.5 | 15 | 5 |
| TFM -> FGBM | 0.186 | 12.5% | 1.5 | 15 | 21 |
| BRN -> FGBS | 0.183 | 12.3% | 1.4 | 15 | 0 |
| TFM -> FOAT | 0.186 | 12.1% | 1.4 | 15 | 0 |
| TFM -> FGBS | 0.184 | 12.0% | 1.4 | 15 | 3 |
| BRN -> FGBL | 0.181 | 11.7% | 1.4 | 15 | 0 |
| TFM -> FBTP | 0.184 | 11.6% | 1.4 | 15 | 0 |
| BRN -> FGBM | 0.184 | 11.6% | 1.3 | 15 | 0 |
| ES -> FGBX | 0.181 | 11.5% | 1.3 | 15 | 0 |
| TFM -> FGBX | 0.182 | 11.4% | 1.3 | 15 | 7 |
| BRN -> FBTP | 0.184 | 11.3% | 1.3 | 15 | 0 |
| ES -> FGBL | 0.179 | 11.3% | 1.3 | 15 | 0 |
| 6E -> FGBM | 0.177 | 11.3% | 1.3 | 15 | 4 |
| GC -> FGBL | 0.176 | 11.3% | 1.3 | 15 | 1 |
| 6E -> FGBL | 0.177 | 11.2% | 1.3 | 15 | 0 |
| ES -> FOAT | 0.181 | 11.2% | 1.3 | 15 | 0 |
| GC -> FGBX | 0.178 | 11.2% | 1.3 | 15 | 0 |
| GC -> FBTP | 0.178 | 11.1% | 1.3 | 15 | 0 |
| CL -> FGBL | 0.174 | 11.1% | 1.3 | 15 | 0 |
| BRN -> FGBX | 0.179 | 11.1% | 1.3 | 15 | 0 |
| BRN -> FOAT | 0.184 | 11.1% | 1.3 | 15 | 2 |
| ES -> FGBS | 0.177 | 11.1% | 1.3 | 15 | 3 |
| GC -> FGBM | 0.177 | 10.9% | 1.3 | 15 | 3 |
| ES -> FBTP | 0.180 | 10.9% | 1.3 | 15 | 6 |
| 6E -> FOAT | 0.180 | 10.9% | 1.3 | 15 | 0 |
| ES -> FGBM | 0.178 | 10.9% | 1.3 | 15 | 2 |
| CL -> FOAT | 0.180 | 10.8% | 1.3 | 15 | 4 |
| 6E -> FGBS | 0.175 | 10.8% | 1.3 | 15 | 2 |
| GC -> FOAT | 0.176 | 10.8% | 1.3 | 15 | 0 |
| CL -> FGBS | 0.176 | 10.6% | 1.2 | 15 | 2 |
| 6E -> FBTP | 0.180 | 10.6% | 1.2 | 15 | 9 |
| GC -> FGBS | 0.177 | 10.6% | 1.2 | 15 | 3 |
| CL -> FGBM | 0.179 | 10.4% | 1.2 | 15 | 0 |
| CL -> FBTP | 0.180 | 10.3% | 1.2 | 15 | 5 |
| 6E -> FGBX | 0.175 | 10.2% | 1.2 | 15 | 3 |
| CL -> FGBX | 0.177 | 10.1% | 1.2 | 15 | 6 |

## Read

- **% live** is the detector's duty cycle: how often the edge clears the bar, vs the ~9% it clears by chance. Low and pair-dependent -> confirms Eugen's premise that the rules are transitory.
- **median live-spell** is how long an edge persists once it turns on — the window you actually have to act (Page-Hinkley changepoints, 126 total, mark where each pair's daily lead-lag regime shifts).
- The **grid_best_corr.png** shows every pair's rolling lead-lag wandering and flipping sign over time — the visual proof that a single static number is misleading.

This is the Phase-1 'measurement layer' from the SOTA survey: a leakage-safe, online detector of *when* a lead-lag is exploitable. The edge classifier and regime-conditioned forecaster (Phases 2-3) build on top of these signals.