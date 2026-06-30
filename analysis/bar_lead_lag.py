"""Bar-level cross-asset lead-lag: do macro/energy/risk drivers lead euro govvies?

Complements analysis/lead_lag.py (which is ms-scale, Eurex-only). Here we work on
OHLCV bars (data/BARS/, see download_bars.py) at 1min / 15min / 1h / 1d to test the
*medium-term* channels — crude->inflation->rates, US rates spillover, risk-on/off —
that don't show up at microstructure scale.

For each bar frequency we build an aligned log-return panel and, for every
(driver, target-bond) pair and lag k>=0, compute corr(r_driver[t], r_bond[t+k]).
k=0 is contemporaneous co-movement; k>0 measures the driver *leading* the bond.

Alignment:
  - intraday (1min/15min/1h): Eurex liquid session 08:00-17:00 Europe/Berlin, where
    all venues trade simultaneously; bars intersected on a common grid.
  - daily: each instrument sampled at its last print <= 17:00 Berlin (Eurex close),
    so daily returns are close(17:00)->close(17:00) on one clock across venues.

Outputs (results/bar_lead_lag/):
    contemp_corr_{freq}.csv / .png   — contemporaneous return correlation matrix
    leadlag_{freq}.csv               — per (driver->bond): best lag (bars) & corr there
    summary.md                       — driver ranking vs each bond, all freqs

Usage:
    python analysis/bar_lead_lag.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BARS_DIR = Path(__file__).resolve().parents[1] / "data" / "BARS"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "bar_lead_lag"

TARGETS = ["FGBS", "FGBM", "FGBL", "FGBX", "FOAT", "FBTP"]          # euro govvies (bonds)
DRIVERS = ["CL", "BRN", "ES", "ZN", "GC", "6E", "TFM"]              # macro / energy / risk
ALL_SYMS = TARGETS + DRIVERS

SESSION_TZ = "Europe/Berlin"
SESSION_OPEN, SESSION_CLOSE = 8, 17
EPS = 1e-12

# freq label -> (pandas resample rule, max forward lag in bars)
FREQS: dict[str, tuple[str, int]] = {
    "1min": ("1min", 30),   # +/- 30 min
    "15min": ("15min", 16),  # +/- 4 h
    "1h": ("1h", 8),         # +/- 8 h
    "1d": ("1d", 5),         # +/- 1 week
}


def load_close(sym: str) -> pd.Series:
    """Load a symbol's 1-min close as a tz-aware (Europe/Berlin) Series."""
    df = pd.read_parquet(BARS_DIR / f"{sym}.parquet", columns=["close"])
    s = df["close"]
    s.index = s.index.tz_convert(SESSION_TZ)
    return s


def return_panel(freq: str) -> pd.DataFrame:
    """Aligned log-return panel [T, n_syms] at the given bar frequency.

    Intraday: restrict to the Eurex liquid session and resample on a common grid.
    Daily: last print at/<=17:00 Berlin per day, then close-to-close returns.
    """
    closes = {s: load_close(s) for s in ALL_SYMS}

    if freq == "1d":
        cols = {}
        for s, ser in closes.items():
            sess = ser[(ser.index.hour < SESSION_CLOSE) | ((ser.index.hour == SESSION_CLOSE) & (ser.index.minute == 0))]
            cols[s] = sess.resample("1d").last()  # last print up to the 17:00 close
        px = pd.DataFrame(cols).dropna(how="all")
    else:
        rule = FREQS[freq][0]
        cols = {}
        for s, ser in closes.items():
            sess = ser[(ser.index.hour >= SESSION_OPEN) & (ser.index.hour < SESSION_CLOSE)]
            cols[s] = sess.resample(rule).last()
        px = pd.DataFrame(cols)

    r = np.log(px).diff()
    r = r.replace([np.inf, -np.inf], np.nan).dropna(how="any")  # common timestamps only
    return r


def zscore(r: pd.DataFrame) -> np.ndarray:
    a = r.values
    return (a - a.mean(0)) / (a.std(0) + EPS)


def lead_lag(r: pd.DataFrame, max_lag: int) -> dict:
    """For each (driver, bond) at lags 0..max_lag: corr(r_driver[t], r_bond[t+k]).

    Returns dict with contemporaneous corr matrix and, per pair, the lag (>=0) with
    the largest |corr| and that corr — the driver's predictive lead on the bond.
    """
    z = zscore(r)  # [T, S]
    cols = list(r.columns)
    di = [cols.index(d) for d in DRIVERS]
    ti = [cols.index(t) for t in TARGETS]
    n = len(z)

    # corr at each lag k: corr(driver[t], bond[t+k]) = mean_t z_d[t]*z_b[t+k]
    best_corr = np.zeros((len(DRIVERS), len(TARGETS)))
    best_lag = np.zeros((len(DRIVERS), len(TARGETS)), dtype=int)
    contemp = np.zeros((len(DRIVERS), len(TARGETS)))
    for a, d in enumerate(di):
        for b, t in enumerate(ti):
            cc = np.array([
                (z[:n - k, d] * z[k:, t]).mean() if k > 0 else (z[:, d] * z[:, t]).mean()
                for k in range(max_lag + 1)
            ])
            contemp[a, b] = cc[0]
            kbest = int(np.abs(cc).argmax())
            best_lag[a, b] = kbest
            best_corr[a, b] = cc[kbest]
    return {"contemp": contemp, "best_lag": best_lag, "best_corr": best_corr,
            "full_corr": np.corrcoef(z.T), "cols": cols, "n": n}


def run_freq(freq: str) -> dict:
    rule, max_lag = FREQS[freq]
    r = return_panel(freq)
    res = lead_lag(r, max_lag)
    res["freq"] = freq
    res["n"] = len(r)

    # contemporaneous full correlation heatmap (all symbols)
    cc = res["full_corr"]
    syms = res["cols"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(syms)), syms, rotation=90)
    ax.set_yticks(range(len(syms)), syms)
    ax.set_title(f"Contemporaneous return corr — {freq} ({res['n']} bars)")
    for i in range(len(syms)):
        for j in range(len(syms)):
            ax.text(j, i, f"{cc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="corr")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"contemp_corr_{freq}.png", dpi=140)
    plt.close(fig)
    pd.DataFrame(cc, index=syms, columns=syms).to_csv(OUT_DIR / f"contemp_corr_{freq}.csv")

    # lead-lag table: best lag (bars) and corr there, driver -> bond
    bl = pd.DataFrame(res["best_lag"], index=DRIVERS, columns=TARGETS)
    bc = pd.DataFrame(res["best_corr"], index=DRIVERS, columns=TARGETS)
    tbl = bc.round(3).astype(str) + " @" + bl.astype(str)
    tbl.to_csv(OUT_DIR / f"leadlag_{freq}.csv")
    return res


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {f: run_freq(f) for f in FREQS}

    lines = ["# Bar-level cross-asset lead-lag (drivers -> euro govvies)\n",
             f"Window 2025-03 -> 2026-06. Drivers: {', '.join(DRIVERS)}.\n",
             "`corr @k`: correlation of driver[t] with bond[t+k]; k>0 = driver leads (bars).\n"]
    for f, res in results.items():
        contemp = pd.DataFrame(res["contemp"], index=DRIVERS, columns=TARGETS)
        bl = pd.DataFrame(res["best_lag"], index=DRIVERS, columns=TARGETS)
        bc = pd.DataFrame(res["best_corr"], index=DRIVERS, columns=TARGETS)
        lines.append(f"\n## {f} grid ({res['n']} bars)\n")
        # rank drivers vs the Bund (FGBL) by |best corr|
        rank = bc["FGBL"].abs().sort_values(ascending=False)
        lines.append("Driver predictive power vs **FGBL (Bund)** — |corr| @ best lag:\n")
        for d in rank.index:
            lines.append(f"- **{d}**: {bc.loc[d,'FGBL']:+.3f} @ lag {bl.loc[d,'FGBL']} | contemp {contemp.loc[d,'FGBL']:+.3f}")
        lines.append("\nBest lead-lag corr (driver↓ → bond→), `corr @lag`:\n")
        lines.append("| driver | " + " | ".join(TARGETS) + " |")
        lines.append("|" + "---|" * (len(TARGETS) + 1))
        for d in DRIVERS:
            cells = [f"{bc.loc[d,t]:+.3f} @{bl.loc[d,t]}" for t in TARGETS]
            lines.append(f"| {d} | " + " | ".join(cells) + " |")
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"Wrote {OUT_DIR}/summary.md")


if __name__ == "__main__":
    main()
