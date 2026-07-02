"""When does the 1s driver->govvie lead switch on? Time-of-day pattern of leadership.

The aggregate 1s study (analysis/cross_lead_lag_1s.py) showed a real but small
driver-leads asymmetry (ZN and CL lead; FESX ~synchronous), concentrated in the first
few seconds. This asks the follow-up Eugen actually wants: is the lead *always weakly
on*, or does it switch on in a pattern — e.g. clustering around the US data window and
cash open, when US rates (ZN) and crude (CL) genuinely drive euro rates?

Method: slide a short window (default 5 min, 1 min step) across each session. In each
window, for every (driver, govvie) pair, measure the *net-lead*
    netlead = sum_{k=1..K} |corr(drv[t], gov[t+k])| - sum_{k=1..K} |corr(gov[t], drv[t+k])|
i.e. how much more the driver leads the bond than the reverse (>0 => driver leads).
Using the difference cancels the per-window noise floor that afflicts a raw |corr|.
Then average netlead by *time of session* (across all days) to expose the pattern.

Times are Europe/Berlin. Reference marks: ~14:30 (US macro releases) and ~15:30
(US cash equity/Treasury open) — the usual suspects for US->EU rates transmission.

Outputs (results/lead_pattern_1s/):
    tod_profile.png   — net-lead vs time-of-session, one line per driver (avg over govvies)
    tod_heatmap.png   — time-of-session x driver->govvie pair net-lead
    tod_profile.csv   — the per-bucket curves
    summary.md        — where leadership concentrates + US-hours share

Usage:
    python analysis/lead_pattern_1s.py [--days 120] [--window 300] [--step 60] [--maxlag 10]
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cross_lead_lag_1s import (DRIVERS, TARGETS, SESSION_OPEN, SESSION_CLOSE,
                               SESSION_TZ, EPS, common_days, _sym_path)

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "lead_pattern_1s"
SESSION_SECONDS = (SESSION_CLOSE - SESSION_OPEN) * 3600
BUCKET_MIN = 15                                    # time-of-session bucket width


def day_return_grid(day: str) -> np.ndarray | None:
    """Regular 1s session log-return grid [SESSION_SECONDS-1, 9] for one day, or None.

    Each symbol's mid is put on the full 08:00-17:00 1s grid and ffilled, so windows
    are exact seconds and comparable across days. Per-symbol winsorized at [0.1, 99.9]%.
    """
    grid = pd.date_range(f"{day} {SESSION_OPEN:02d}:00:00", periods=SESSION_SECONDS,
                         freq="1s", tz=SESSION_TZ)
    cols = []
    for sym in DRIVERS + TARGETS:
        f = _sym_path(sym, day)
        if not f.exists():
            return None
        df = pd.read_parquet(f, columns=["ts_event", "bid_px_00", "ask_px_00"])
        scale = 1e-9 if df["bid_px_00"].abs().max() > 1e6 else 1.0
        mid = (df["bid_px_00"] + df["ask_px_00"]) * scale / 2
        t = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert(SESSION_TZ)
        s = pd.Series(mid.values, index=t).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        g = s.reindex(grid, method="ffill")
        if g.notna().sum() < SESSION_SECONDS // 2:
            return None
        cols.append(g)
    px = pd.DataFrame(dict(zip(DRIVERS + TARGETS, cols))).ffill().bfill()
    r = np.log(px).diff().iloc[1:].to_numpy().copy()
    if not np.isfinite(r).all():
        return None
    lo, hi = np.percentile(r, 0.1, axis=0), np.percentile(r, 99.9, axis=0)
    np.clip(r, lo, hi, out=r)
    return r


def _norm(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(0)) / (a.std(0) + EPS)


def window_netlead(win: np.ndarray, n_drv: int, max_lag: int) -> np.ndarray:
    """Net-lead [n_drv, n_tgt] for one window: |driver-leads| - |bond-leads| summed over k."""
    drv, tgt = win[:, :n_drv], win[:, n_drv:]
    n = len(win)
    fwd = np.zeros((n_drv, tgt.shape[1]))   # driver leads
    rev = np.zeros((n_drv, tgt.shape[1]))   # bond leads
    for k in range(1, max_lag + 1):
        a, b = _norm(drv[:n - k]), _norm(tgt[k:])
        fwd += np.abs(a.T @ b / (n - k))
        c, d = _norm(tgt[:n - k]), _norm(drv[k:])
        rev += np.abs((c.T @ d / (n - k)).T)   # -> [n_drv, n_tgt]
    return fwd - rev


def run(n_days: int, window: int, step: int, max_lag: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    days = common_days(n_days)
    n_drv, n_tgt = len(DRIVERS), len(TARGETS)
    n_buckets = SESSION_SECONDS // (BUCKET_MIN * 60)
    print(f"lead pattern 1s: {len(days)} days ({days[0]} -> {days[-1]}), "
          f"{window}s window / {step}s step, lags 1-{max_lag}s")

    acc = np.zeros((n_buckets, n_drv, n_tgt))
    cnt = np.zeros(n_buckets)
    used = 0
    for day in days:
        r = day_return_grid(day)
        if r is None:
            continue
        T = len(r)
        for start in range(0, T - window + 1, step):
            win = r[start:start + window]
            if win.std(0).min() < EPS:
                continue
            bucket = min(start // (BUCKET_MIN * 60), n_buckets - 1)
            acc[bucket] += window_netlead(win, n_drv, max_lag)
            cnt[bucket] += 1
        used += 1
        if used % 20 == 0:
            print(f"  {used} days done (through {day})")
    if used == 0:
        raise RuntimeError("no usable days")
    prof = acc / np.maximum(cnt[:, None, None], 1)      # [buckets, n_drv, n_tgt] mean net-lead
    _write_outputs(prof, n_buckets, used, window, step, max_lag)
    print(f"  wrote outputs -> {OUT_DIR}")


def _bucket_times() -> list[str]:
    out = []
    for i in range(SESSION_SECONDS // (BUCKET_MIN * 60)):
        mins = SESSION_OPEN * 60 + i * BUCKET_MIN
        out.append(f"{mins // 60:02d}:{mins % 60:02d}")
    return out


def _write_outputs(prof: np.ndarray, n_buckets: int, used: int,
                   window: int, step: int, max_lag: int) -> None:
    times = _bucket_times()
    by_driver = prof.mean(axis=2)                       # [buckets, n_drv], avg over govvies

    df = pd.DataFrame(by_driver, index=times, columns=DRIVERS)
    df.index.name = "session_time"
    df.to_csv(OUT_DIR / "tod_profile.csv")

    # time-of-day profile per driver
    x = np.arange(n_buckets)
    fig, ax = plt.subplots(figsize=(11, 5))
    for di, d in enumerate(DRIVERS):
        ax.plot(x, by_driver[:, di], marker="o", ms=3, label=d)
    ax.axhline(0, color="k", lw=0.5)
    for hh, mm, lab in [(14, 30, "US data ~14:30"), (15, 30, "US open ~15:30")]:
        bx = ((hh - SESSION_OPEN) * 60 + mm) / BUCKET_MIN
        ax.axvline(bx, color="tab:red", ls="--", lw=1)
        ax.text(bx, ax.get_ylim()[1], lab, rotation=90, va="top", ha="right", fontsize=7, color="tab:red")
    ax.set_xticks(x[::4], [times[i] for i in x[::4]], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("time of session (Europe/Berlin)")
    ax.set_ylabel("net-lead  (|driver leads| - |bond leads|)")
    ax.set_title(f"When does each driver lead the euro govvies? 1s net-lead by time of day "
                 f"({used} days)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tod_profile.png", dpi=140)
    plt.close(fig)

    # heatmap: time-of-day x pair
    pairs = [f"{d}->{t}" for d in DRIVERS for t in TARGETS]
    mat = prof.reshape(n_buckets, -1).T                 # [pairs, buckets]
    fig, ax = plt.subplots(figsize=(12, 6))
    vmax = np.abs(mat).max() + EPS
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(pairs)), pairs, fontsize=6)
    ax.set_xticks(x[::4], [times[i] for i in x[::4]], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("time of session (Europe/Berlin)")
    ax.set_title(f"Net-lead by time of day and pair (red = driver leads), {used} days")
    fig.colorbar(im, ax=ax, label="net-lead")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tod_heatmap.png", dpi=140)
    plt.close(fig)

    _summary_md(by_driver, times, used, window, step, max_lag)


def _summary_md(by_driver: np.ndarray, times: list[str], used: int,
                window: int, step: int, max_lag: int) -> None:
    # US window = 14:00-17:00 Berlin
    us_mask = np.array([t >= "14:00" for t in times])
    lines = [
        "# 1s driver->govvie leadership — time-of-day pattern\n",
        f"{used} days, {window}s window / {step}s step, lead lags 1-{max_lag}s. "
        "net-lead = |driver leads| - |bond leads|, averaged over the six govvies per driver.\n",
        "\n## Peak leadership time per driver\n",
        "| driver | peak time | peak net-lead | mean 08-14 | mean 14-17 (US) | US/EU ratio |",
        "|---|---|---|---|---|---|",
    ]
    for di, d in enumerate(DRIVERS):
        col = by_driver[:, di]
        peak = int(np.argmax(col))
        eu = col[~us_mask].mean()
        us = col[us_mask].mean()
        ratio = us / eu if eu > EPS else float("nan")
        lines.append(f"| {d} | {times[peak]} | {col[peak]:+.4f} | {eu:+.4f} | {us:+.4f} | "
                     f"{ratio:.2f} |")
    lines += [
        "\n## Read\n",
        "- A driver whose net-lead is flat across the day has no timing pattern — its (weak) "
        "lead is always-on background.",
        "- A driver that spikes in the 14:00-17:00 window (US data + cash open) is leading euro "
        "rates specifically when the US is active — the exploitable, tellable pattern.",
        "- See **tod_profile.png** (lines) and **tod_heatmap.png** (per-pair).",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--window", type=int, default=300, help="window (seconds)")
    p.add_argument("--step", type=int, default=60, help="step (seconds)")
    p.add_argument("--maxlag", type=int, default=10, help="max lead tested (seconds)")
    args = p.parse_args()
    run(args.days, args.window, args.step, args.maxlag)


if __name__ == "__main__":
    main()
