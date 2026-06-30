"""Tushar's linear next-tick model (L1 subset) vs TLOB, on HIS next-tick label.

Unlike `tushar_linear_baseline.py` (which scored his features against our 1s
SMOOTHED_MID_TIME target), this replicates his ACTUAL labelling method and scores
the same head-to-head on it:

  target = sign of the next *settled* one-tick mid move (LabelingType.NEXT_TICK)
           — binary up/down, only 1-tick-spread events, NO d_max horizon cap
           (his full-data / no-horizon-cap evaluation; accuracy to beat ~76%).
  split  = chronological 80 / 5 / 15 (train / val-skipped / test), Bund.

Feature families (40 of his 90; f4 depth dropped — needs L10 books we lack):
  f1 notional imbalance (Bund)            1
  f2 net trade intensity (5 instr, 5s)    5
  f3 TOB-VWAP EWMA-diff (5 instr x 5 hl)  25
  f5 lead-lag vs Bund (4 instr)            4
  f6 move-from-open 07:00 LDN (5 instr)    5

Run from the tlob-eurex repo root with the shared interpreter.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "/home/rig/PythonProjects/EurexNextTickSignalModeling/src")
import instruments_eurex  # noqa: F401
from tlob import constants as cst
from tlob.data.dataloader import DataLoader
from tlob.data.utils_data import labeling
from nt_signals.ewma_vec import ewma_norm_ct, intensity_ct  # HIS exact EWMA math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

INSTR = {"bund": "FGBL", "bobl": "FGBM", "buxl": "FGBX", "btp": "FBTP", "oat": "FOAT"}
TICK = {"bund": 0.01, "bobl": 0.01, "buxl": 0.02, "btp": 0.01, "oat": 0.01}
HLS = [1, 5, 10, 30, 60]
HL_VOL = 5.0
LN2 = np.log(2.0)
WARMUP_LDN = "07:00"
TUSHAR_REF = 0.76   # his full-data (no horizon cap) sign accuracy to beat


def build_signals(sym: str, tick: float) -> pd.DataFrame:
    """Per-day (file) event-time signals from TBBO: mid, TOB-VWAP EWMA-diffs (f3),
    trade-intensity tvi_buy/sell (f2 source). EWMA resets each day."""
    frames = []
    for f in sorted(glob.glob(f"data/{sym}_TBBO/*.parquet")):
        d = pd.read_parquet(f, columns=["ts_event", "bid_px_00", "ask_px_00",
                                        "bid_sz_00", "ask_sz_00", "side", "size"])
        d = d.dropna(subset=["bid_px_00", "ask_px_00"])
        d = d[(d["bid_sz_00"] > 0) & (d["ask_sz_00"] > 0)]
        t = pd.DatetimeIndex(d["ts_event"])
        ldn = t.tz_convert("Europe/London")
        d = d[ldn.time >= pd.Timestamp(WARMUP_LDN).time()]              # warm from 07:00 LDN
        if len(d) < 2:
            continue
        d = d.sort_values("ts_event", kind="stable")
        ts_ns = pd.DatetimeIndex(d["ts_event"]).asi8
        ts_s = ts_ns / 1e9
        b0 = d["bid_px_00"].to_numpy(); a0 = d["ask_px_00"].to_numpy()
        bs0 = d["bid_sz_00"].to_numpy(float); as0 = d["ask_sz_00"].to_numpy(float)
        mid = 0.5 * (b0 + a0)
        tob_vwap = (b0 * bs0 + a0 * as0) / (bs0 + as0)
        side = d["side"].to_numpy(); size = d["size"].to_numpy(float)
        out = {"date": f.split("/")[-1][:10], "ts_us": ts_ns // 1000, "mid": mid,
               "tvi_buy": intensity_ct(ts_s, size * (side == "B"), HL_VOL),
               "tvi_sell": intensity_ct(ts_s, size * (side == "A"), HL_VOL)}
        for hl in HLS:
            out[f"f3_hl{hl}"] = (tob_vwap - ewma_norm_ct(ts_s, tob_vwap, hl)) / tick
        frames.append(pd.DataFrame(out))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    # ---- 1. target frame + NEXT_TICK label (TLOB-identical) ---------------- #
    dl = DataLoader(
        horizon=60, initial_seq_size=256, data_sources=["FGBL_TBBO"], add_features=False,
        labeling_type=cst.LabelingType.NEXT_TICK, join_type=cst.JoinType.LEFT,
        len_trade_window=10, n_lob_levels=1, num_classes=3, fee=0.01,
        experiment_type=cst.ExperimentType.TRAINING, add_trades=True, only_trades=False,
        is_databento=True, filter_session=True,
    )
    g = dl.load_data_databento()[0].sort_values("transact_time").reset_index(drop=True)
    g = dl._filter_to_session(g)
    g["transact_time"] = g["transact_time"].astype("int64") // 10**3
    g = g[[c for c in g.columns if c != "transact_time"] + ["transact_time"]]
    lab = labeling(g, 1, cst.LabelingType.NEXT_TICK, 3, ["FGBL_TBBO"], 0.01, 10,
                   tick_size=0.01)
    g["label"] = lab
    grid = g.dropna(subset=["label"]).reset_index(drop=True)
    grid["ts_us"] = grid["transact_time"].astype("int64")
    grid["date"] = pd.to_datetime(grid["ts_us"], unit="us", utc=True).dt.strftime("%Y-%m-%d")
    bn = grid["FGBL_TBBO_bid1_price"] * grid["FGBL_TBBO_bid1_quantity"]
    an = grid["FGBL_TBBO_ask1_price"] * grid["FGBL_TBBO_ask1_quantity"]
    grid["f1_imbalance"] = ((bn - an) / (bn + an)).to_numpy()
    cls, cnt = np.unique(grid["label"], return_counts=True)
    print(f"[grid] labeled Bund settled events: {len(grid):,}  days: {grid['date'].nunique()}  "
          f"label dist: {dict(zip(cls.astype(int), (cnt / cnt.sum()).round(4)))}")

    # ---- 2. per-instrument signals ---------------------------------------- #
    sig = {k: build_signals(v, TICK[k]) for k, v in INSTR.items()}
    for k in sig:
        print(f"[signals] {k}: {len(sig[k]):,} rows")

    # ---- 3. assemble features per day (daily reset for f5/f6) -------------- #
    def asof(left_ts, right, vcols):
        L = pd.DataFrame({"ts_us": np.asarray(left_ts, dtype="int64")})
        m = pd.merge_asof(L, right[["ts_us"] + vcols].sort_values("ts_us"),
                          on="ts_us", direction="backward")
        return m

    rows = []
    for date, gd in grid.groupby("date", sort=True):
        gd = gd.sort_values("ts_us")
        ts = gd["ts_us"].to_numpy()
        feat = {"f1_imbalance": gd["f1_imbalance"].to_numpy()}
        day_sig = {k: sig[k][sig[k]["date"] == date] for k in INSTR}
        # bund mid-move times this day (for lead-lag reset reference)
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
            # f2: decay-project the as-of intensity to the grid time
            ti_ts = asof(ts, sk.assign(ti_ts=sk["ts_us"]), ["ti_ts"])["ti_ts"]
            dt = np.where(ti_ts.notna(), (ts - ti_ts.fillna(0).to_numpy()) / 1e6, np.inf)
            net = (m["tvi_buy"].to_numpy() - m["tvi_sell"].to_numpy()) * np.exp(-LN2 / HL_VOL * np.clip(dt, 0, None))
            feat[f'f2_{k}'] = np.nan_to_num(net, posinf=0.0, neginf=0.0)
            # f6: move from this day's open mid
            feat[f"f6_{k}"] = (np.nan_to_num(m['mid'].to_numpy(), nan=sk['mid'].iloc[0], posinf=sk['mid'].iloc[0], neginf=sk['mid'].iloc[0]) - sk["mid"].iloc[0]) / TICK[k]
            # f5: predictor cum move since last bund move (k != bund)
            if k != "bund":
                mid_now = m["mid"].to_numpy()
                mid_ref = asof(last_move, sk, ["mid"])["mid"].to_numpy()
                feat[f'f5_{k}'] = np.nan_to_num((mid_now - mid_ref) / TICK[k], posinf=0.0, neginf=0.0)
        feat["label"] = gd["label"].to_numpy()
        feat["ts_us"] = ts
        rows.append(pd.DataFrame(feat))
    X = pd.concat(rows, ignore_index=True).sort_values("ts_us").reset_index(drop=True)
    feat_cols = [c for c in X.columns if c not in ("label", "ts_us")]
    X.to_parquet("results/tushar_nexttick_features.parquet", index=False)
    print(f"[features] {len(feat_cols)} cols, {len(X):,} rows -> cached")

    # ---- 4. chronological 80/5/15 split (mirror TLOB SPLIT) ---------------- #
    # Binary sign target: 0 = up, 2 = down (3-class container, flat never present).
    n = len(X); tr_end = int(0.80 * n); te_start = int(0.85 * n)
    variants = {
        "f1 only (imbalance)": ["f1_imbalance"],
        "f1+f3 (imbalance+momentum)": ["f1_imbalance"] + [c for c in feat_cols if c.startswith("f3_")],
        "all 40 (his L1 set)": feat_cols,
    }
    ytr = X.iloc[:tr_end]["label"].to_numpy().astype(int)
    yte = X.iloc[te_start:]["label"].to_numpy().astype(int)
    print(f"\n[linear baseline] train {len(ytr):,} / test {len(yte):,}   "
          f"Tushar full-data ref: sign acc ~{TUSHAR_REF:.2f}")
    print(f"{'features':>30s} {'signAcc':>8s} {'macroF1':>8s} {'upF1':>6s} {'downF1':>7s}")
    for name, cols in variants.items():
        sc = StandardScaler().fit(X.iloc[:tr_end][cols].to_numpy())
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(X.iloc[:tr_end][cols].to_numpy()), ytr)
        pred = clf.predict(sc.transform(X.iloc[te_start:][cols].to_numpy()))
        acc = accuracy_score(yte, pred)
        per = f1_score(yte, pred, labels=[0, 2], average=None)
        mac = f1_score(yte, pred, labels=[0, 2], average="macro")
        print(f"{name:>30s} {acc:>8.4f} {mac:>8.4f} {per[0]:>6.3f} {per[1]:>7.3f}")


if __name__ == "__main__":
    main()
