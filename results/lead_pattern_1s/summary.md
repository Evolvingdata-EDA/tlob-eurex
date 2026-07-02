# 1s driver->govvie leadership — time-of-day pattern

231 days, 300s window / 60s step, lead lags 1-10s. net-lead = |driver leads| - |bond leads|, averaged over the six govvies per driver.


## Peak leadership time per driver

| driver | peak time | peak net-lead | mean 08-14 | mean 14-17 (US) | US/EU ratio |
|---|---|---|---|---|---|
| CL | 15:30 | +0.0185 | +0.0091 | +0.0107 | 1.18 |
| FESX | 11:45 | +0.0058 | -0.0011 | -0.0007 | nan |
| ZN | 16:00 | +0.0257 | +0.0108 | +0.0145 | 1.35 |

## Read

- A driver whose net-lead is flat across the day has no timing pattern — its (weak) lead is always-on background.
- A driver that spikes in the 14:00-17:00 window (US data + cash open) is leading euro rates specifically when the US is active — the exploitable, tellable pattern.
- See **tod_profile.png** (lines) and **tod_heatmap.png** (per-pair).