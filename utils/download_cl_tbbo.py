"""Download continuous front-month TBBO for CL (WTI crude, NYMEX/Globex), GLBX.MDP3.

CL is a candidate cross-asset predictor source for the euro-govvie next-tick models
(Tushar's suggestion, alongside FESX). Continuous front-month uses `.n.0` (calendar
front) on GLBX — same convention as the ZN util. Window matches the Eurex TBBO farms
(2025-06-01 -> 2026-05-29) so the stacked multi-source builder shares the window.

Writes data/CL.n.0/{date}_tbbo.parquet; symlink data/CL_TBBO -> CL.n.0 after.
Resume-safe: existing per-day parquet is skipped.

Run: /home/rig/PythonProjects/tlob-a2a/.venv/bin/python utils/download_cl_tbbo.py
"""
import os

from tlob.data.databento.download_databento import download_databento_data

DATASET = "GLBX.MDP3"
SYMBOL = "CL.n.0"          # continuous front-month, calendar roll
SCHEMA = "tbbo"
START_DATE = "2025-07-01"  # inside the Standard-plan trailing-12-month free window
END_DATE = "2026-05-29"    # match Eurex TBBO window (also inside free window)
OUTPUT_DIR = "data"        # -> data/CL.n.0/{date}_tbbo.parquet


def main() -> None:
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set")
    print(f"=== {SYMBOL} TBBO {START_DATE}..{END_DATE} ({DATASET}) ===", flush=True)
    download_databento_data(
        api_key=api_key, dataset=DATASET, symbol=SYMBOL,
        start_date=START_DATE, end_date=END_DATE, output_dir=OUTPUT_DIR,
        stype_in="continuous", prefer_batch=False, schema=SCHEMA,
    )
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
