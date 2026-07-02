# 1s cross-asset lead-lag — do CL / FESX / ZN lead the euro govvies at 1-60s?

TBBO top-of-book, 1s mids, session 08:00-17:00 Berlin, 231 trading days (-> 2026-05-29). C(k) = corr(driver[t], govvie[t+k]); k>0 = driver leads.

Per-lag noise band ~ +/-0.0008 (2 SE); a peak must clear this to be real.


**Strongest lead:** ZN -> FOAT, corr +0.023 at +1s (vs contemporaneous +0.072).


## Per pair (sorted by peak leading corr)

`lead` = driver leads (k>0); `lag` = govvie leads (k<0). corr @ lag(s).

| driver -> govvie | contemp (k=0) | peak lead (k>0) | peak govvie-leads (k<0) |
|---|---|---|---|
| ZN -> FOAT | +0.072 | +0.023 @ +1s | +0.009 @ -1s |
| ZN -> FGBL | +0.102 | +0.022 @ +1s | +0.013 @ -1s |
| ZN -> FBTP | +0.077 | +0.022 @ +1s | +0.010 @ -1s |
| CL -> FGBL | -0.039 | -0.021 @ +1s | -0.008 @ -1s |
| CL -> FBTP | -0.038 | -0.021 @ +1s | -0.007 @ -1s |
| ZN -> FGBX | +0.072 | +0.021 @ +1s | +0.009 @ -1s |
| CL -> FOAT | -0.033 | -0.021 @ +1s | -0.007 @ -1s |
| CL -> FGBX | -0.023 | -0.016 @ +1s | -0.005 @ -1s |
| ZN -> FGBM | +0.066 | +0.015 @ +1s | +0.008 @ -1s |
| CL -> FGBM | -0.024 | -0.014 @ +1s | -0.007 @ -1s |
| ZN -> FGBS | +0.046 | +0.012 @ +1s | +0.006 @ -1s |
| FESX -> FBTP | +0.033 | +0.011 @ +1s | +0.010 @ -1s |
| CL -> FGBS | -0.018 | -0.011 @ +1s | -0.004 @ -1s |
| FESX -> FOAT | +0.026 | +0.010 @ +1s | +0.006 @ -1s |
| FESX -> FGBL | +0.018 | +0.004 @ +1s | +0.007 @ -1s |
| FESX -> FGBX | +0.011 | +0.003 @ +1s | +0.003 @ -1s |
| FESX -> FGBM | +0.013 | +0.003 @ +1s | +0.003 @ -1s |
| FESX -> FGBS | +0.011 | +0.002 @ +1s | +0.002 @ -1s |

## Read

- If the **contemp (k=0)** column dominates and the k>0 peak is barely above the noise band, the relationship is synchronous, not a tradeable lead (same story as 1-min).
- A **k>0 peak well above the band and above the k<0 side** is a genuine driver lead; its lag is the seconds you have to react within.
- See **ccf_grid.png** — a lead looks like a bump on the shaded (green) right side.