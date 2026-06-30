"""Download June-2026 FGBL (Bund) EOBI into the a2a canonical source tree.

Extends the existing FGBL.v.0 history (which ends 2026-05-29) with the June
window Tushar actually evaluated on. mbp-10 (10-level depth) for the depth
check + tbbo (L1) as the matched-window control. Continuous front (.v.0,
volume roll) handles the 2026-06-05 roll automatically.

Cost verified $0.00 via get_cost (trailing-12mo free window). Files land in
tlob-a2a/data/XEUR.EOBI/FGBL.v.0/{date}_{mbp10,tbbo}.parquet, where the symlink
farms read from. Per-day, skips existing.
"""
import os

from tlob.data.databento import download_databento_data

DATASET = "XEUR.EOBI"
SYMBOL = "FGBL.v.0"
OUTPUT_DIR = "/home/rig/PythonProjects/tlob-a2a/data/XEUR.EOBI"
START_DATE = "2026-06-01"
END_DATE = "2026-06-28"   # avail end 2026-06-29; inclusive day loop


def main() -> None:
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set")
    for schema in ("tbbo", "mbp-10"):   # tbbo first (tiny), then depth
        print(f"\n=== {SYMBOL} {schema} {START_DATE}..{END_DATE} ===", flush=True)
        download_databento_data(
            api_key=api_key, dataset=DATASET, symbol=SYMBOL,
            start_date=START_DATE, end_date=END_DATE, output_dir=OUTPUT_DIR,
            stype_in="continuous", prefer_batch=False, schema=schema,
        )
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
