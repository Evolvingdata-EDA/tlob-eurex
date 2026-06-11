"""Order-book lead-lag analysis across the six Eurex govvie futures.

For every ordered pair (A, B) and lag k, computes corr(r_A[t], r_B[t+k]) on a
fixed time grid — "does A's move now predict B's move k steps ahead?" — and
aggregates per-day correlations into a lead matrix. The net-lead score
L[A,B] - L[B,A] ranks who moves first; row means rank overall leadership.

Outputs (results/lead_lag/):
    lead_matrix_{grid}.csv   — summed cross-corr L[i,j] = sum_k corr(rA, rB shifted k)
    net_lead_{grid}.csv      — antisymmetric net-lead matrix L - L.T
    best_lag_{grid}.csv      — argmax-corr lag (in grid steps) per ordered pair
    heatmap_{grid}.png       — net-lead heatmap
    summary.md               — leader ranking + per-pair best lag/corr, both grids

Usage:
    python analysis/lead_lag.py [--days 60]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SYMBOLS = ["FGBS", "FGBM", "FGBL", "FGBX", "FBTP", "FOAT"]  # short → long duration, then periphery
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "XEUR.EOBI"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "lead_lag"
LAST_DAY = "2026-05-29"  # last day all six symbols share
SESSION_TZ = "Europe/Berlin"
SESSION_OPEN, SESSION_CLOSE = 8, 17
# grid label -> (pandas freq, max lag in steps): 100ms/±5s and 1s/±30s
GRIDS: dict[str, tuple[str, int]] = {"100ms": ("100ms", 50), "1s": ("1s", 30)}
EPS = 1e-12


def common_days(n_days: int) -> list[str]:
    """Return the last `n_days` dates (YYYY-MM-DD) present for ALL symbols."""
    per_sym = []
    for sym in SYMBOLS:
        files = sorted((DATA_ROOT / f"{sym}.v.0").glob("*_tbbo.parquet"))
        per_sym.append({f.name.split("_")[0] for f in files})
    days = sorted(set.intersection(*per_sym))
    days = [d for d in days if d <= LAST_DAY]
    return days[-n_days:]


def day_returns(day: str, freq: str) -> np.ndarray | None:
    """Standardized session-hours mid returns for all symbols on one day.

    Returns:
        Array [T, n_symbols] of z-scored grid returns, or None if any symbol
        has no usable data that day.
    """
    cols = []
    idx = None
    for sym in SYMBOLS:
        f = DATA_ROOT / f"{sym}.v.0" / f"{day}_tbbo.parquet"
        df = pd.read_parquet(f, columns=["ts_event", "bid_px_00", "ask_px_00"])
        scale = 1e-9 if df["bid_px_00"].abs().max() > 1e6 else 1.0
        mid = (df["bid_px_00"] + df["ask_px_00"]) * scale / 2
        t = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert(SESSION_TZ)
        s = pd.Series(mid.values, index=t)
        s = s[(s.index.hour >= SESSION_OPEN) & (s.index.hour < SESSION_CLOSE)]
        if len(s) < 100:
            return None
        g = s.resample(freq).last().ffill()
        cols.append(g)
        idx = g.index if idx is None else idx.intersection(g.index)
    if idx is None or len(idx) < 1000:
        return None
    mat = np.column_stack([c.reindex(idx).ffill().values for c in cols])  # [T, S]
    r = np.diff(np.log(mat), axis=0)  # [T-1, S]
    r = r[~np.isnan(r).any(axis=1)]
    if len(r) < 1000:
        return None
    r = (r - r.mean(axis=0)) / (r.std(axis=0) + EPS)  # z-score per symbol
    return r


def day_lead_tensors(r: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlation tensor for one day.

    Args:
        r: Standardized returns [T, S].
        max_lag: Number of forward lags.

    Returns:
        (corr [max_lag, S, S], counts) where corr[k-1, i, j] = corr(r_i[t], r_j[t+k]).
    """
    n_sym = r.shape[1]
    out = np.zeros((max_lag, n_sym, n_sym))
    for k in range(1, max_lag + 1):
        a, b = r[:-k], r[k:]  # [T-k, S]
        out[k - 1] = a.T @ b / len(a)  # z-scored cols -> this IS the correlation
    return out, np.ones(max_lag)


def run_grid(days: list[str], label: str, freq: str, max_lag: int) -> dict:
    """Aggregate lead tensors across days for one grid; write CSVs + heatmap."""
    acc, n_used = None, 0
    for day in days:
        r = day_returns(day, freq)
        if r is None:
            print(f"[{label}] {day}: skipped (insufficient data)")
            continue
        t, _ = day_lead_tensors(r, max_lag)
        acc = t if acc is None else acc + t
        n_used += 1
        if n_used % 10 == 0:
            print(f"[{label}] {n_used} days done (through {day})")
    if acc is None:
        raise RuntimeError(f"[{label}] no usable days")
    corr = acc / n_used  # mean per-lag cross-corr [K, S, S]

    lead = corr.sum(axis=0)  # L[i,j] = summed corr of i's move with j's future moves
    np.fill_diagonal(lead, 0.0)
    net = lead - lead.T
    best_lag = corr.argmax(axis=0) + 1  # in grid steps
    best_corr = corr.max(axis=0)

    sym = SYMBOLS
    pd.DataFrame(lead, index=sym, columns=sym).to_csv(OUT_DIR / f"lead_matrix_{label}.csv")
    pd.DataFrame(net, index=sym, columns=sym).to_csv(OUT_DIR / f"net_lead_{label}.csv")
    pd.DataFrame(best_lag, index=sym, columns=sym).to_csv(OUT_DIR / f"best_lag_{label}.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = np.abs(net).max() + EPS
    im = ax.imshow(net, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(sym)), sym)
    ax.set_yticks(range(len(sym)), sym)
    ax.set_title(f"Net lead (row leads column), {label} grid, {n_used} days")
    ax.set_xlabel("lagging instrument")
    ax.set_ylabel("leading instrument")
    for i in range(len(sym)):
        for j in range(len(sym)):
            ax.text(j, i, f"{net[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="sum_k corr(r_i, r_j(+k)) - reverse")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"heatmap_{label}.png", dpi=150)
    plt.close(fig)

    return {"label": label, "n_days": n_used, "lead": lead, "net": net,
            "best_lag": best_lag, "best_corr": best_corr}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=60, help="trading days to use")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    days = common_days(args.days)
    print(f"Using {len(days)} common days: {days[0]} → {days[-1]}")

    results = [run_grid(days, label, freq, k) for label, (freq, k) in GRIDS.items()]

    lines = [f"# Eurex govvie lead-lag — {days[0]} → {days[-1]} ({len(days)} days)\n"]
    for res in results:
        net, sym = res["net"], SYMBOLS
        ranking = sorted(zip(sym, net.mean(axis=1)), key=lambda x: -x[1])
        lines.append(f"\n## Grid {res['label']} ({res['n_days']} days used)\n")
        lines.append("Leadership ranking (mean net-lead vs all others):\n")
        for s, v in ranking:
            lines.append(f"- **{s}**: {v:+.3f}")
        lines.append("\nPer-pair (i leads j): best lag (steps) / corr at best lag:\n")
        lines.append("| leads ↓ / lags → | " + " | ".join(sym) + " |")
        lines.append("|" + "---|" * (len(sym) + 1))
        for i, si in enumerate(sym):
            cells = [
                "—" if i == j else f"{res['best_lag'][i, j]} / {res['best_corr'][i, j]:.3f}"
                for j in range(len(sym))
            ]
            lines.append(f"| {si} | " + " | ".join(cells) + " |")
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"Wrote {OUT_DIR}/summary.md")


if __name__ == "__main__":
    main()
