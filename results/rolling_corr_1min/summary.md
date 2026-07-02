# Simple 1-minute rolling lead-lag correlation (no train/test)

120m window, 15m step, 9100 windows per pair. Direct correlation on each full window — no out-of-sample split.


**Noise reference** (unrelated series, ~119 points): a null correlation has E|corr| ~ **0.073** and 2-sigma ~ **0.183**. Read the averages against these.


## Per pair — lead k = 1 min (unbiased) and best-lag (selection-inflated)

| driver -> govvie | mean corr (k=1) | mean \|corr\| (k=1) | % \|corr\|>2s | mean \|corr\| best-lag |
|---|---|---|---|---|
| ZN -> FGBL | -0.000 | 0.085 | 8.5% | 0.188 |
| ZN -> FBTP | +0.002 | 0.084 | 8.5% | 0.186 |
| TFM -> FGBM | -0.006 | 0.083 | 8.1% | 0.186 |
| ZN -> FGBS | +0.012 | 0.082 | 7.1% | 0.185 |
| ZN -> FGBM | +0.007 | 0.082 | 7.7% | 0.188 |
| ZN -> FOAT | +0.006 | 0.082 | 7.9% | 0.186 |
| TFM -> FOAT | -0.001 | 0.081 | 7.5% | 0.186 |
| ZN -> FGBX | +0.008 | 0.081 | 7.0% | 0.189 |
| TFM -> FGBS | -0.005 | 0.081 | 7.3% | 0.184 |
| BRN -> FGBS | -0.011 | 0.081 | 7.2% | 0.183 |
| ES -> FGBL | -0.004 | 0.079 | 6.7% | 0.179 |
| 6E -> FGBM | +0.005 | 0.079 | 6.1% | 0.177 |
| ES -> FGBS | -0.005 | 0.079 | 6.6% | 0.177 |
| ES -> FBTP | +0.003 | 0.079 | 7.3% | 0.180 |
| BRN -> FGBM | -0.012 | 0.079 | 6.8% | 0.184 |
| BRN -> FOAT | -0.011 | 0.079 | 6.6% | 0.184 |
| TFM -> FGBX | -0.006 | 0.079 | 6.5% | 0.182 |
| 6E -> FBTP | -0.002 | 0.078 | 6.7% | 0.180 |
| GC -> FGBX | -0.001 | 0.078 | 6.0% | 0.178 |
| BRN -> FBTP | -0.006 | 0.078 | 6.8% | 0.184 |
| CL -> FBTP | -0.010 | 0.078 | 6.6% | 0.180 |
| ES -> FGBX | -0.003 | 0.078 | 6.6% | 0.181 |
| ES -> FOAT | +0.000 | 0.078 | 7.1% | 0.181 |
| BRN -> FGBX | -0.005 | 0.077 | 6.5% | 0.179 |
| GC -> FGBM | +0.007 | 0.077 | 6.1% | 0.177 |
| TFM -> FGBL | -0.002 | 0.077 | 6.5% | 0.185 |
| CL -> FGBS | -0.013 | 0.077 | 6.2% | 0.176 |
| TFM -> FBTP | -0.004 | 0.077 | 7.1% | 0.184 |
| CL -> FGBM | -0.012 | 0.077 | 6.0% | 0.179 |
| CL -> FOAT | -0.011 | 0.077 | 5.6% | 0.180 |
| 6E -> FGBL | +0.001 | 0.077 | 6.4% | 0.177 |
| GC -> FBTP | +0.004 | 0.077 | 6.0% | 0.178 |
| 6E -> FOAT | +0.003 | 0.077 | 6.5% | 0.180 |
| 6E -> FGBX | +0.002 | 0.077 | 5.6% | 0.175 |
| GC -> FGBS | +0.007 | 0.077 | 5.6% | 0.177 |
| GC -> FOAT | +0.001 | 0.076 | 6.1% | 0.176 |
| CL -> FGBX | -0.008 | 0.076 | 5.5% | 0.177 |
| 6E -> FGBS | +0.005 | 0.076 | 5.4% | 0.175 |
| BRN -> FGBL | -0.003 | 0.076 | 6.2% | 0.181 |
| ES -> FGBM | -0.002 | 0.075 | 5.7% | 0.178 |
| GC -> FGBL | +0.002 | 0.075 | 5.1% | 0.176 |
| CL -> FGBL | -0.005 | 0.074 | 5.4% | 0.174 |

## Read

- **mean corr (k=1)** is the plain signed lead-lag correlation, averaged over windows — no selection, so it is directly comparable to the ~0.073 noise level.
- **mean |corr| best-lag** picks the strongest of leads 1..10 each window, so it sits above the noise floor by construction (selection bias).
- **% |corr|>2sigma** is how often a single window's correlation clears chance (~0.183); at pure noise this would be ~5%.