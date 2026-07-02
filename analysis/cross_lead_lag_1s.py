"""1-second cross-asset lead-lag: do CL / FESX / ZN lead the euro govvies at 1-60s?

The 1-min bar study (analysis/rolling_lead_lag.py) found that once contemporaneous
(same-bar) co-movement is removed, the tradeable cross-asset lead into the govvies is
thin (strongest ~1.6x the noise floor). The hypothesis is that a genuine lead lives at
*seconds* and is blurred inside a 1-min bar. This script tests that directly on TBBO
top-of-book at 1s resolution.

For every (driver, govvie) pair we estimate the cross-correlation function
    C(k) = corr(r_driver[t], r_govvie[t+k]),  k in [-K, K] seconds,
averaged over trading days. k>0 means the driver *leads* the bond by k seconds; k<0
means the bond leads. A clean lead shows up as a peak at some k>0 that towers over the
k<0 side and over the |corr| you'd see by chance. The peak lag is the horizon you'd
have to react within.

Drivers with tick data: CL (WTI), FESX (Euro STOXX 50), ZN (US 10Y). Targets: the six
Eurex govvie futures. Session 08:00-17:00 Berlin, mid = (bid+ask)/2, 1s grid.

Outputs (results/cross_lead_lag_1s/):
    ccf.csv            — full C(k) curve per pair
    ccf_grid.png       — 3x6 small-multiples of C(k), lead side shaded
    peak_corr.png      — driver x govvie: corr at the best leading lag (k>0)
    best_lag.png       — driver x govvie: the leading lag (s) at that peak
    summary.md         — per-pair contemp vs peak-lead corr, lead horizon, lead vs lag

Usage:
    python analysis/cross_lead_lag_1s.py [--days 60] [--maxlag 60]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "cross_lead_lag_1s"

DRIVER_DIRS = {"CL": "CL.n.0", "FESX": "FESX.v.0", "ZN": "ZN.n.0"}   # tick-data drivers
DRIVERS = list(DRIVER_DIRS)
TARGETS = ["FGBS", "FGBM", "FGBL", "FGBX", "FBTP", "FOAT"]           # euro govvies
LAST_DAY = "2026-05-29"                                             # all 9 share up to here

SESSION_TZ = "Europe/Berlin"
SESSION_OPEN, SESSION_CLOSE = 8, 17
FREQ = "1s"
EPS = 1e-12


def _sym_path(sym: str, day: str) -> Path:
    sub = DRIVER_DIRS[sym] if sym in DRIVER_DIRS else f"XEUR.EOBI/{sym}.v.0"
    return ROOT / sub / f"{day}_tbbo.parquet"


def common_days(n_days: int) -> list[str]:
    """Last n_days dates present for ALL drivers and targets, up to LAST_DAY."""
    sets = []
    for sym in DRIVERS + TARGETS:
        d = _sym_path(sym, "x").parent
        sets.append({f.name.split("_")[0] for f in d.glob("*_tbbo.parquet")})
    days = sorted(d for d in set.intersection(*sets) if d <= LAST_DAY)
    return days[-n_days:]


def day_returns(day: str) -> np.ndarray | None:
    """Z-scored 1s session mid-returns for all 9 symbols on one day -> [T, 9] or None.

    Mids on a common 1s grid (session hours), log-returns, per-symbol winsorized at
    [0.1%, 99.9%] to kill stray ticks, then z-scored so a matmul gives correlation.
    """
    cols, idx = [], None
    for sym in DRIVERS + TARGETS:
        f = _sym_path(sym, day)
        if not f.exists():
            return None
        df = pd.read_parquet(f, columns=["ts_event", "bid_px_00", "ask_px_00"])
        scale = 1e-9 if df["bid_px_00"].abs().max() > 1e6 else 1.0
        mid = (df["bid_px_00"] + df["ask_px_00"]) * scale / 2
        t = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert(SESSION_TZ)
        s = pd.Series(mid.values, index=t)
        s = s[(s.index.hour >= SESSION_OPEN) & (s.index.hour < SESSION_CLOSE)]
        if len(s) < 500:
            return None
        g = s.resample(FREQ).last().ffill()
        cols.append(g)
        idx = g.index if idx is None else idx.intersection(g.index)
    if idx is None or len(idx) < 5000:
        return None
    mat = np.column_stack([c.reindex(idx).ffill().to_numpy() for c in cols])   # [T, 9]
    r = np.diff(np.log(mat), axis=0)
    r = r[~np.isnan(r).any(axis=1)]
    if len(r) < 5000:
        return None
    lo = np.percentile(r, 0.1, axis=0)
    hi = np.percentile(r, 99.9, axis=0)
    np.clip(r, lo, hi, out=r)
    return (r - r.mean(0)) / (r.std(0) + EPS)


def day_ccf(r: np.ndarray, max_lag: int) -> np.ndarray:
    """Cross-correlation function C[k, d, t] for k in [-K, K] on one day's returns.

    C[k] = corr(r_driver[u], r_target[u+k]); k>0 -> driver leads target by k seconds.
    r is z-scored, so the shifted dot product over the overlap is the correlation.
    """
    n_d, n_t = len(DRIVERS), len(TARGETS)
    drv, tgt = r[:, :n_d], r[:, n_d:]
    lags = range(-max_lag, max_lag + 1)
    out = np.zeros((len(lags), n_d, n_t))
    n = len(r)
    for i, k in enumerate(lags):
        if k >= 0:
            a, b = drv[:n - k], tgt[k:]          # driver[u] vs target[u+k]
        else:
            a, b = drv[-k:], tgt[:n + k]         # driver[u] vs target[u+k], k<0
        m = a.shape[0]
        a = (a - a.mean(0)) / (a.std(0) + EPS)
        b = (b - b.mean(0)) / (b.std(0) + EPS)
        out[i] = a.T @ b / m
    return out


def run(n_days: int, max_lag: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    days = common_days(n_days)
    print(f"1s cross lead-lag: {len(days)} days ({days[0]} -> {days[-1]}), lags +/-{max_lag}s")

    acc, used = None, 0
    for day in days:
        r = day_returns(day)
        if r is None:
            continue
        c = day_ccf(r, max_lag)
        acc = c if acc is None else acc + c
        used += 1
        if used % 10 == 0:
            print(f"  {used} days done (through {day})")
    if acc is None:
        raise RuntimeError("no usable days")
    ccf = acc / used                                          # [2K+1, n_d, n_t]
    lags = np.arange(-max_lag, max_lag + 1)

    # noise floor: |corr| that a single 1s cross-corr clears by chance, ~T-per-day pts
    # averaged over `used` days -> std ~ 1/sqrt(used * T). We report the per-lag SE.
    se = 1.0 / np.sqrt(used * _median_T(days))
    _write_outputs(ccf, lags, max_lag, used, se)
    print(f"  wrote outputs -> {OUT_DIR} (per-lag SE ~ {se:.4f})")


def _median_T(days: list[str]) -> float:
    return 30000.0     # ~ session seconds with a print; used only for the SE reference


def _write_outputs(ccf: np.ndarray, lags: np.ndarray, max_lag: int, used: int, se: float) -> None:
    n_d, n_t = len(DRIVERS), len(TARGETS)
    pos = lags > 0                                            # driver-leads side
    neg = lags < 0
    k0 = np.where(lags == 0)[0][0]

    rows, curves = [], {}
    for di, d in enumerate(DRIVERS):
        for ti, t in enumerate(TARGETS):
            c = ccf[:, di, ti]
            curves[f"{d}->{t}"] = c
            lead_ix = np.where(pos)[0][np.abs(c[pos]).argmax()]
            lag_ix = np.where(neg)[0][np.abs(c[neg]).argmax()]
            rows.append({
                "driver": d, "target": t,
                "contemp_corr": float(c[k0]),
                "peak_lead_corr": float(c[lead_ix]),      # driver leads (k>0)
                "peak_lead_lag_s": int(lags[lead_ix]),
                "peak_lag_corr": float(c[lag_ix]),        # bond leads (k<0)
                "peak_lag_lag_s": int(lags[lag_ix]),
            })
    summ = pd.DataFrame(rows)
    pd.DataFrame(curves, index=lags).to_csv(OUT_DIR / "ccf.csv", index_label="lag_s")

    # 3x6 CCF grid
    fig, axes = plt.subplots(n_d, n_t, figsize=(2.2 * n_t, 1.8 * n_d), sharex=True, sharey=True)
    for di, d in enumerate(DRIVERS):
        for ti, t in enumerate(TARGETS):
            ax = axes[di, ti]
            ax.plot(lags, ccf[:, di, ti], lw=0.8)
            ax.axvline(0, color="k", lw=0.4)
            ax.axhline(0, color="k", lw=0.3)
            ax.axhspan(-se * 2, se * 2, color="grey", alpha=0.25)   # +/-2 SE noise band
            ax.axvspan(0, max_lag, color="tab:green", alpha=0.05)   # driver-leads region
            if di == 0:
                ax.set_title(t, fontsize=8)
            if ti == 0:
                ax.set_ylabel(d, fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"1s cross-corr C(k)=corr(driver[t], govvie[t+k]); k>0 = driver leads "
                 f"({used} days)", fontsize=10)
    fig.supxlabel("lag k (seconds)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ccf_grid.png", dpi=140)
    plt.close(fig)

    for col, fname, title, fmt in [
        ("peak_lead_corr", "peak_corr.png", "Corr at best leading lag (driver leads, k>0)", "{:.3f}"),
        ("peak_lead_lag_s", "best_lag.png", "Leading lag at peak (seconds)", "{:.0f}"),
    ]:
        mat = summ.pivot(index="driver", columns="target", values=col).reindex(index=DRIVERS, columns=TARGETS)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        im = ax.imshow(mat.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(n_t), TARGETS)
        ax.set_yticks(range(n_d), DRIVERS)
        ax.set_title(title)
        for i in range(n_d):
            for j in range(n_t):
                ax.text(j, i, fmt.format(mat.values[i, j]), ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=140)
        plt.close(fig)

    _summary_md(summ, used, max_lag, se)


def _summary_md(summ: pd.DataFrame, used: int, max_lag: int, se: float) -> None:
    top = summ.reindex(summ["peak_lead_corr"].abs().sort_values(ascending=False).index)
    b = top.iloc[0]
    lines = [
        f"# 1s cross-asset lead-lag — do CL / FESX / ZN lead the euro govvies at 1-{max_lag}s?\n",
        f"TBBO top-of-book, 1s mids, session 08:00-17:00 Berlin, {used} trading days "
        f"(-> {LAST_DAY}). C(k) = corr(driver[t], govvie[t+k]); k>0 = driver leads.\n",
        f"Per-lag noise band ~ +/-{2 * se:.4f} (2 SE); a peak must clear this to be real.\n",
        f"\n**Strongest lead:** {b.driver} -> {b.target}, corr {b.peak_lead_corr:+.3f} at "
        f"+{b.peak_lead_lag_s}s (vs contemporaneous {b.contemp_corr:+.3f}).\n",
        "\n## Per pair (sorted by peak leading corr)\n",
        "`lead` = driver leads (k>0); `lag` = govvie leads (k<0). corr @ lag(s).\n",
        "| driver -> govvie | contemp (k=0) | peak lead (k>0) | peak govvie-leads (k<0) |",
        "|---|---|---|---|",
    ]
    for _, r in top.iterrows():
        lines.append(f"| {r.driver} -> {r.target} | {r.contemp_corr:+.3f} | "
                     f"{r.peak_lead_corr:+.3f} @ +{r.peak_lead_lag_s}s | "
                     f"{r.peak_lag_corr:+.3f} @ {r.peak_lag_lag_s}s |")
    lines += [
        "\n## Read\n",
        "- If the **contemp (k=0)** column dominates and the k>0 peak is barely above the "
        "noise band, the relationship is synchronous, not a tradeable lead (same story as "
        "1-min).",
        "- A **k>0 peak well above the band and above the k<0 side** is a genuine driver "
        "lead; its lag is the seconds you have to react within.",
        "- See **ccf_grid.png** — a lead looks like a bump on the shaded (green) right side.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=60, help="trading days to use (most recent)")
    p.add_argument("--maxlag", type=int, default=60, help="max lead/lag tested (seconds)")
    args = p.parse_args()
    run(args.days, args.maxlag)


if __name__ == "__main__":
    main()
