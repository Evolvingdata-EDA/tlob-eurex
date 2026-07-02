"""Tushar 40-feature LINEAR model (multinomial logistic regression) on the
2-tick SMOOTHED_MID pure-sign target, emitted to a maker-sim result.csv and run
through the passive market-making simulation.

Reuses analysis/tushar_linear_baseline.py's exact feature builder; the only
changes are: target = event-horizon h=2 (SMOOTHED_MID, label_threshold_ticks=0),
and we carry the book + aggressor-signed trade tape per row so maker_sim can fill.
Run from the tlob-eurex repo root with the shared interpreter.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "/home/rig/PythonProjects/EurexNextTickSignalModeling/src")
import instruments_eurex  # noqa: F401
from tlob import constants as cst
from tlob.data.dataloader import DataLoader
from tlob.data.utils_data import labeling
from tlob.backtest.maker_sim import run_maker_sim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analysis.tushar_linear_baseline import build_signals, INSTR, TICK, HLS, LN2, HL_VOL

HORIZON = 2
OUTDIR = "checkpoints/TLOB/FGBL_TBBO_linear40_smid_h2_maker"


def main() -> None:
    dl = DataLoader(
        horizon=HORIZON, initial_seq_size=256, data_sources=["FGBL_TBBO"], add_features=False,
        labeling_type=cst.LabelingType.SMOOTHED_MID, join_type=cst.JoinType.LEFT,
        len_trade_window=10, n_lob_levels=1, num_classes=3, fee=0.01,
        experiment_type=cst.ExperimentType.TRAINING, add_trades=True, only_trades=False,
        is_databento=True, filter_session=True,
    )
    g = dl.load_data_databento()[0].sort_values("transact_time").reset_index(drop=True)
    g = dl._filter_to_session(g)
    g["transact_time"] = g["transact_time"].astype("int64") // 10**3
    g = g[[c for c in g.columns if c != "transact_time"] + ["transact_time"]]
    lab = labeling(g, HORIZON, cst.LabelingType.SMOOTHED_MID, 3, ["FGBL_TBBO"], 0.01, 10,
                   label_threshold_ticks=0.0, tick_size=0.01)
    g["label"] = lab
    grid = g.dropna(subset=["label"]).reset_index(drop=True)
    grid["ts_us"] = grid["transact_time"].astype("int64")
    grid["date"] = pd.to_datetime(grid["ts_us"], unit="us", utc=True).dt.strftime("%Y-%m-%d")
    bn = grid["FGBL_TBBO_bid1_price"] * grid["FGBL_TBBO_bid1_quantity"]
    an = grid["FGBL_TBBO_ask1_price"] * grid["FGBL_TBBO_ask1_quantity"]
    grid["f1_imbalance"] = ((bn - an) / (bn + an)).to_numpy()
    print(f"[grid] labeled Bund events: {len(grid):,}  days: {grid['date'].nunique()}")

    sig = {k: build_signals(v, TICK[k]) for k, v in INSTR.items()}

    def asof(left_ts, right, vcols):
        L = pd.DataFrame({"ts_us": np.asarray(left_ts, dtype="int64")})
        return pd.merge_asof(L, right[["ts_us"] + vcols].sort_values("ts_us"),
                             on="ts_us", direction="backward")

    # book + aggressor-signed trade tape carried per row for the maker sim.
    CARRY = ["ask1_price", "bid1_price", "ask1_qty", "bid1_qty",
             "ask_exec_qty", "bid_exec_qty", "price", "label", "ts_us"]
    rows = []
    for date, gd in grid.groupby("date", sort=True):
        gd = gd.sort_values("ts_us")
        ts = gd["ts_us"].to_numpy()
        feat = {"f1_imbalance": gd["f1_imbalance"].to_numpy()}
        day_sig = {k: sig[k][sig[k]["date"] == date] for k in INSTR}
        sb = day_sig["bund"]
        bm = sb.loc[sb["mid"].ne(sb["mid"].shift()), "ts_us"]
        last_move = pd.merge_asof(pd.DataFrame({"ts_us": ts}),
                                  pd.DataFrame({"ts_us": bm.to_numpy(), "bm_ts": bm.to_numpy()}).sort_values("ts_us"),
                                  on="ts_us", direction="backward")["bm_ts"]
        last_move = last_move.fillna(ts[0]).to_numpy().astype("int64")
        for k in INSTR:
            sk = day_sig[k]
            if len(sk) == 0:
                for h in HLS:
                    feat[f"f3_{k}_hl{h}"] = np.zeros(len(ts))
                feat[f"f2_{k}"] = np.zeros(len(ts)); feat[f"f6_{k}"] = np.zeros(len(ts))
                if k != "bund":
                    feat[f"f5_{k}"] = np.zeros(len(ts))
                continue
            m = asof(ts, sk, [f"f3_hl{h}" for h in HLS] + ["tvi_buy", "tvi_sell", "mid"])
            for h in HLS:
                feat[f"f3_{k}_hl{h}"] = np.nan_to_num(m[f'f3_hl{h}'].to_numpy(), posinf=0.0, neginf=0.0)
            ti_ts = asof(ts, sk.assign(ti_ts=sk["ts_us"]), ["ti_ts"])["ti_ts"]
            dt = np.where(ti_ts.notna(), (ts - ti_ts.fillna(0).to_numpy()) / 1e6, np.inf)
            net = (m["tvi_buy"].to_numpy() - m["tvi_sell"].to_numpy()) * np.exp(-LN2 / HL_VOL * np.clip(dt, 0, None))
            feat[f'f2_{k}'] = np.nan_to_num(net, posinf=0.0, neginf=0.0)
            feat[f"f6_{k}"] = (np.nan_to_num(m['mid'].to_numpy(), nan=sk['mid'].iloc[0], posinf=sk['mid'].iloc[0], neginf=sk['mid'].iloc[0]) - sk["mid"].iloc[0]) / TICK[k]
            if k != "bund":
                mid_now = m["mid"].to_numpy()
                mid_ref = asof(last_move, sk, ["mid"])["mid"].to_numpy()
                feat[f'f5_{k}'] = np.nan_to_num((mid_now - mid_ref) / TICK[k], posinf=0.0, neginf=0.0)
        # carry book + trade tape for the maker sim
        ap = gd["FGBL_TBBO_ask1_price"].to_numpy(); bp = gd["FGBL_TBBO_bid1_price"].to_numpy()
        q = gd["FGBL_TBBO_quantity"].to_numpy(float); s = gd["FGBL_TBBO_side"].to_numpy(float)
        feat["ask1_price"] = ap; feat["bid1_price"] = bp
        feat["ask1_qty"] = gd["FGBL_TBBO_ask1_quantity"].to_numpy()
        feat["bid1_qty"] = gd["FGBL_TBBO_bid1_quantity"].to_numpy()
        feat["ask_exec_qty"] = np.where(s > 0, q, 0.0)   # buy-aggressor lifts ask
        feat["bid_exec_qty"] = np.where(s < 0, q, 0.0)   # sell-aggressor hits bid
        feat["price"] = (ap + bp) / 2.0
        feat["label"] = gd["label"].to_numpy()
        feat["ts_us"] = ts
        rows.append(pd.DataFrame(feat))
    X = pd.concat(rows, ignore_index=True).sort_values("ts_us").reset_index(drop=True)
    feat_cols = [c for c in X.columns if c not in CARRY]
    print(f"[features] {len(feat_cols)} feature cols, {len(X):,} rows")

    # chronological 80/5/15 split (mirror the framework); fit on train, predict test.
    n = len(X); tr_end = int(0.80 * n); te_start = int(0.85 * n)
    ytr = X.iloc[:tr_end]["label"].to_numpy().astype(int)
    sc = StandardScaler().fit(X.iloc[:tr_end][feat_cols].to_numpy())
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(sc.transform(X.iloc[:tr_end][feat_cols].to_numpy()), ytr)
    te = X.iloc[te_start:].reset_index(drop=True)
    pred = clf.predict(sc.transform(te[feat_cols].to_numpy())).astype(int)

    from sklearn.metrics import f1_score, accuracy_score
    yte = te["label"].to_numpy().astype(int)
    print(f"[linear h2] test acc {accuracy_score(yte, pred):.4f}  macroF1 {f1_score(yte, pred, average='macro'):.4f}")

    os.makedirs(OUTDIR, exist_ok=True)
    res = pd.DataFrame({
        "Preds": pred, "label": yte,
        "price": te["price"].to_numpy(),
        "ask1_price": te["ask1_price"].to_numpy(), "bid1_price": te["bid1_price"].to_numpy(),
        "ask1_qty": te["ask1_qty"].to_numpy().astype("int64"),
        "bid1_qty": te["bid1_qty"].to_numpy().astype("int64"),
        "ask_exec_qty": te["ask_exec_qty"].to_numpy().astype("int64"),
        "bid_exec_qty": te["bid_exec_qty"].to_numpy().astype("int64"),
        "timestamp": pd.to_datetime(te["ts_us"].to_numpy(), unit="us"),
    })
    res.to_csv(os.path.join(OUTDIR, "result.csv"), index=False)
    print(f"[result] wrote {len(res):,} test rows -> {OUTDIR}/result.csv")

    run_maker_sim(OUTDIR, label="linear40 h2", symbol="FGBL_TBBO", horizon=HORIZON,
                  cancel_advances_queue=True)


if __name__ == "__main__":
    main()
