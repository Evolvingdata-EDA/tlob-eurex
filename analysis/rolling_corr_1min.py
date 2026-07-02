"""Simple 1-minute rolling lead-lag correlation (no train/test split).

A stripped-down companion to analysis/rolling_lead_lag.py: for each rolling window
and each (driver, govvie) pair, just compute the lead-lag correlation directly on the
whole window — corr(driver[t], govvie[t+k]) — and average it. No out-of-sample split,
no "live" classifier.

Two views per pair:
  - fixed lead k = 1 min (unbiased: no lag selection),
  - best of leads k = 1..maxlag (|corr|-maximising; inflated by lag selection).

Reference noise level: for two unrelated series over n~window points, a Pearson
correlation has E|r| ~ sqrt(2 / (pi * n)) and 2-sigma ~ 2 / sqrt(n). We report both so
the averages are readable against chance.

Outputs (results/rolling_corr_1min/):
    per_pair.csv   — mean signed / mean |corr| / %|corr|>2sigma, per pair, both views
    summary.md     — table + the noise reference

Usage:
    python analysis/rolling_corr_1min.py [--window 120] [--step 15] [--maxlag 10] [--fixed 1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rolling_lead_lag import (DRIVERS, TARGETS, EPS, load_session_closes,
                              daily_return_panels, _zscore, _corr_at_lag)

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "rolling_corr_1min"


def run(window: int, step: int, max_lag: int, fixed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = daily_return_panels(load_session_closes())
    n_drv, n_tgt = len(DRIVERS), len(TARGETS)

    # running accumulators, shape [n_drv, n_tgt]
    sum_fixed = np.zeros((n_drv, n_tgt))       # signed corr at k=fixed
    sumabs_fixed = np.zeros((n_drv, n_tgt))    # |corr| at k=fixed
    sumabs_best = np.zeros((n_drv, n_tgt))     # best |corr| over k=1..max_lag
    n_win = 0
    n_eff = window - fixed                      # points entering the fixed-lag corr
    sigma = 1.0 / np.sqrt(n_eff)                # SE of a null correlation
    thr = 2 * sigma
    above_fixed = np.zeros((n_drv, n_tgt))      # count |corr@fixed| > 2 sigma

    for _, r in panels:
        for start in range(0, len(r) - window + 1, step):
            win = r[start:start + window]
            if win.std(0).min() < EPS:
                continue
            zd, zt = _zscore(win[:, :n_drv]), _zscore(win[:, n_drv:])
            cc = np.stack([_corr_at_lag(zd, zt, k) for k in range(1, max_lag + 1)])  # [K,D,T]
            cf = cc[fixed - 1]                                                       # corr at k=fixed
            sum_fixed += cf
            sumabs_fixed += np.abs(cf)
            sumabs_best += np.abs(cc).max(0)
            above_fixed += (np.abs(cf) > thr)
            n_win += 1

    mean_signed = sum_fixed / n_win
    mean_abs = sumabs_fixed / n_win
    mean_abs_best = sumabs_best / n_win
    pct_above = 100 * above_fixed / n_win

    rows = []
    for di, d in enumerate(DRIVERS):
        for ti, t in enumerate(TARGETS):
            rows.append({"driver": d, "target": t,
                         "mean_corr_k%d" % fixed: mean_signed[di, ti],
                         "mean_abs_k%d" % fixed: mean_abs[di, ti],
                         "pct_gt_2sigma": pct_above[di, ti],
                         "mean_abs_bestlag": mean_abs_best[di, ti]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_pair.csv", index=False)

    e_abs_null = np.sqrt(2 / (np.pi * n_eff))
    _summary_md(df, n_win, window, step, max_lag, fixed, e_abs_null, thr)
    print(f"{n_win} windows/pair. noise: E|corr|~{e_abs_null:.3f}, 2sigma~{thr:.3f}")
    print(df.sort_values("mean_abs_k%d" % fixed, ascending=False).head(8).to_string(index=False))


def _summary_md(df, n_win, window, step, max_lag, fixed, e_abs_null, thr):
    top = df.sort_values(f"mean_abs_k{fixed}", ascending=False)
    lines = [
        "# Simple 1-minute rolling lead-lag correlation (no train/test)\n",
        f"{window}m window, {step}m step, {n_win} windows per pair. Direct correlation on "
        f"each full window — no out-of-sample split.\n",
        f"\n**Noise reference** (unrelated series, ~{window-fixed} points): a null correlation "
        f"has E|corr| ~ **{e_abs_null:.3f}** and 2-sigma ~ **{thr:.3f}**. Read the averages "
        f"against these.\n",
        f"\n## Per pair — lead k = {fixed} min (unbiased) and best-lag (selection-inflated)\n",
        f"| driver -> govvie | mean corr (k={fixed}) | mean \\|corr\\| (k={fixed}) | "
        f"% \\|corr\\|>2s | mean \\|corr\\| best-lag |",
        "|---|---|---|---|---|",
    ]
    for _, r in top.iterrows():
        lines.append(f"| {r.driver} -> {r.target} | {r[f'mean_corr_k{fixed}']:+.3f} | "
                     f"{r[f'mean_abs_k{fixed}']:.3f} | {r.pct_gt_2sigma:.1f}% | "
                     f"{r.mean_abs_bestlag:.3f} |")
    lines += [
        "\n## Read\n",
        f"- **mean corr (k={fixed})** is the plain signed lead-lag correlation, averaged over "
        f"windows — no selection, so it is directly comparable to the ~{e_abs_null:.3f} noise level.",
        "- **mean |corr| best-lag** picks the strongest of leads 1.."
        f"{max_lag} each window, so it sits above the noise floor by construction (selection bias).",
        f"- **% |corr|>2sigma** is how often a single window's correlation clears chance "
        f"(~{thr:.3f}); at pure noise this would be ~5%.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window", type=int, default=120)
    p.add_argument("--step", type=int, default=15)
    p.add_argument("--maxlag", type=int, default=10)
    p.add_argument("--fixed", type=int, default=1, help="fixed lead (minutes) for the unbiased corr")
    args = p.parse_args()
    run(args.window, args.step, args.maxlag, args.fixed)


if __name__ == "__main__":
    main()
