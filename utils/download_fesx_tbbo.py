"""Download continuous front-month TBBO for FESX (Euro Stoxx 50, Eurex), XEUR.EOBI.

FESX is a candidate cross-asset predictor source for the euro-govvie next-tick models
(Tushar's suggestion, alongside CL). Continuous front-month uses `.v.0` (volume roll)
on XEUR — same convention as the Eurex govvie farms (FGBL.v.0 etc.). Window matches
the Eurex TBBO farms (2025-06-01 -> 2026-05-29) so the stacked builder shares the window.

Note: XEUR.EOBI history starts 2025-03-10, so this window is the practical max overlap
with the govvie sources anyway.

Writes data/FESX.v.0/{date}_tbbo.parquet; symlink data/FESX_TBBO -> FESX.v.0 after.
Resume-safe: existing per-day parquet is skipped.

Run: /home/rig/PythonProjects/tlob-a2a/.venv/bin/python utils/download_fesx_tbbo.py
"""
import os

from tlob.data.databento.download_databento import download_databento_data

DATASET = "XEUR.EOBI"
SYMBOL = "FESX.v.0"        # continuous front-month, volume roll (Eurex convention)
SCHEMA = "tbbo"
START_DATE = "2025-07-01"  # inside the Standard-plan trailing-12-month free window
END_DATE = "2026-05-29"    # match Eurex TBBO window (also inside free window)
OUTPUT_DIR = "data"        # -> data/FESX.v.0/{date}_tbbo.parquet


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
