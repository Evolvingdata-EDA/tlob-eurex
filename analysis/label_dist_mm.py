"""Market-making label distribution for FGBL (Bund), smoothed directional drift.

Uses the new SMOOTHED_MID_TIME `label_threshold_ticks` gate (absolute, exact):
    0.0 -> pure sign(delta)   (any up/down drift in the smoothed mid)
    0.5 -> half-tick
    1.0 -> one tick
Faithful to training: real DataLoader load + session gate, real labeling().
Run from the tlob-eurex repo root with the shared interpreter.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
import instruments_eurex  # noqa: F401  (registers FGBL/... economics)
from tlob import constants as cst
from tlob.data.dataloader import DataLoader
from tlob.data.utils_data import labeling

SOURCE = "FGBL_TBBO"
TICK = 0.01
HORIZONS = (1, 3, 5)        # seconds
THRESHOLDS = (("sign (0.0t)", 0.0), ("half (0.5t)", 0.5), ("one (1.0t)", 1.0))


def load_session_frame():
    dl = DataLoader(
        horizon=60, initial_seq_size=256, data_sources=[SOURCE], add_features=False,
        labeling_type=cst.LabelingType.SMOOTHED_MID_TIME, join_type=cst.JoinType.LEFT,
        len_trade_window=10, n_lob_levels=1, num_classes=3, fee=0.01,
        experiment_type=cst.ExperimentType.TRAINING, add_trades=True, only_trades=False,
        is_databento=True, filter_session=True,
    )
    df = dl.load_data_databento()[0].sort_values("transact_time").reset_index(drop=True)
    df = dl._filter_to_session(df)
    df["transact_time"] = df["transact_time"].astype("int64") // 10**3
    return df[[c for c in df.columns if c != "transact_time"] + ["transact_time"]]


def main() -> None:
    df = load_session_frame()
    print(f"\n===== FGBL (Bund) TBBO  session rows: {len(df):,} =====\n")
    hdr = f"{'h(s)':>4s} {'target':>14s} {'n':>11s} {'up%':>7s} {'flat%':>7s} {'down%':>7s} {'dir%':>6s}"
    print(hdr)
    for h in HORIZONS:
        for name, thr in THRESHOLDS:
            lab = labeling(df, h, cst.LabelingType.SMOOTHED_MID_TIME, 3, [SOURCE], 0.01, 10,
                           label_threshold_ticks=thr, tick_size=TICK)
            lab = lab[~np.isnan(lab)].astype(int)
            n = len(lab)
            up, flat, dn = 100 * np.bincount(lab, minlength=3) / n
            print(f"{h:>4d} {name:>14s} {n:>11,} {up:>7.2f} {flat:>7.2f} {dn:>7.2f} {up + dn:>6.2f}")
        print()


if __name__ == "__main__":
    main()
