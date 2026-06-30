"""Download OHLCV-1m bars for the cross-asset lead-lag study.

Targets are the six Eurex govvie futures; drivers are the macro/energy/risk
instruments that plausibly lead euro rates at bar (1m-1week) horizons:
crude (CL/Brent), equity (ES), US rates (ZN), gold (GC), EUR/USD (6E),
and TTF gas (TFM). All pulled as `ohlcv-1m` continuous front-month (`.v.0`),
which is tiny (~$0) and uniform across venues.

Output: data/BARS/{SYMBOL}.parquet — columns [open, high, low, close, volume],
DatetimeIndex `ts_event` in UTC, plus a `symbol` column.

Usage:
    python analysis/download_bars.py [--start 2025-03-10] [--end 2026-06-25]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import databento as db
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "BARS"

# symbol -> databento dataset. Continuous front-month (.v.0, volume-rolled).
SYMBOLS: dict[str, str] = {
    # --- targets: Eurex govvie futures (XEUR.EOBI) ---
    "FGBS": "XEUR.EOBI",  # Schatz  2Y
    "FGBM": "XEUR.EOBI",  # Bobl    5Y
    "FGBL": "XEUR.EOBI",  # Bund    10Y
    "FGBX": "XEUR.EOBI",  # Buxl    30Y
    "FOAT": "XEUR.EOBI",  # OAT     FR 10Y
    "FBTP": "XEUR.EOBI",  # BTP     IT 10Y
    # --- drivers ---
    "CL": "GLBX.MDP3",    # WTI crude  (CME)
    "ES": "GLBX.MDP3",    # S&P 500    (CME)
    "ZN": "GLBX.MDP3",    # 10Y UST    (CME)
    "GC": "GLBX.MDP3",    # gold       (CME)
    "6E": "GLBX.MDP3",    # EUR/USD    (CME)
    "BRN": "IFEU.IMPACT",  # Brent crude (ICE Futures Europe)
    "TFM": "NDEX.IMPACT",  # TTF gas     (ICE Endex)
}


def download(symbol: str, dataset: str, start: str, end: str) -> int:
    """Fetch ohlcv-1m for one continuous symbol and write a parquet. Returns rows."""
    client = db.Historical()
    data = client.timeseries.get_range(
        dataset=dataset,
        symbols=[f"{symbol}.v.0"],
        schema="ohlcv-1m",
        stype_in="continuous",
        start=start,
        end=end,
    )
    df = data.to_df()  # DatetimeIndex ts_event (UTC); cols open/high/low/close/volume/symbol
    if df.empty:
        print(f"  {symbol:5s} EMPTY")
        return 0
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df["symbol"] = symbol
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / f"{symbol}.parquet")
    print(f"  {symbol:5s} {dataset:12s} rows={len(df):>7d}  {df.index[0].date()} -> {df.index[-1].date()}")
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-03-10", help="bonds availability start on XEUR")
    parser.add_argument("--end", default="2026-06-25")
    args = parser.parse_args()

    print(f"Downloading ohlcv-1m {args.start} -> {args.end} into {OUT_DIR}")
    total = 0
    for sym, ds in SYMBOLS.items():
        total += download(sym, ds, args.start, args.end)
    print(f"Done. {total} total rows across {len(SYMBOLS)} symbols.")


if __name__ == "__main__":
    main()
