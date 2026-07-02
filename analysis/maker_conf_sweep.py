"""Confidence-threshold sweep for the h1 TLOB maker: only pull the toxic side
when the predicted-class probability >= tau; else quote both sides. tau=0 = full
pull (current signal), tau>=1 = never pull (baseline). Finds whether an
intermediate tau beats both.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import instruments_eurex  # noqa
from tlob import instruments
from tlob.backtest.maker_sim import _simulate
from tlob.backtest.run_backtesting import _filter_to_session

D = "checkpoints/TLOB/FGBL_TBBO_labeling_type_smoothed_mid_horizon_1_num_features_8_is_3d_False_fee_0.01__fgbl_smid_h1"
SYM = "FGBL_TBBO"

raw = pd.read_csv(D + "/result.csv").sort_values("timestamp").reset_index(drop=True)
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
spec = instruments.get_spec(SYM); sess = instruments.session_params(SYM)
raw = _filter_to_session(raw.set_index("timestamp"), sess.tz, sess.open_hour, sess.close_hour).reset_index()
day_id = raw["timestamp"].dt.tz_localize("UTC").dt.tz_convert(sess.tz).dt.strftime("%Y%m%d").astype("int64").to_numpy()
pred = raw["Preds"].to_numpy("int64")
probs = raw[["prob_0", "prob_1", "prob_2"]].to_numpy("float64")
conf = probs[np.arange(len(pred)), pred]   # prob of the predicted class

base = dict(
    bid_px=raw["bid1_price"].to_numpy("float64"), ask_px=raw["ask1_price"].to_numpy("float64"),
    bid_qd=raw["bid1_qty"].to_numpy("float64"), ask_qd=raw["ask1_qty"].to_numpy("float64"),
    bid_exec=raw["bid_exec_qty"].to_numpy("float64"), ask_exec=raw["ask_exec_qty"].to_numpy("float64"),
    mid=raw["price"].to_numpy("float64"), pred=pred, day_id=day_id,
    point_value=spec.point_value, maker_fee=0.0, taker_fee=spec.fee_per_side_native,
    tick=spec.tick_size, inv_cap=50, clip=1, cancel_advances_queue=True,
)
print(f"conf (pred-class prob): min {conf.min():.2f} median {np.median(conf):.2f} max {conf.max():.2f}")
b = _simulate(use_signal=False, **base)
print(f"\n{'variant':>16} {'net EUR':>11} {'Sharpe':>8} {'fills':>9} {'adverse EUR':>12} {'pull%':>7}")
print(f"{'baseline':>16} {b.net_pnl_eur:>11,.0f} {b.sharpe_net_ann:>8.2f} {b.n_bid_fills+b.n_ask_fills:>9,} {b.adverse_selection_eur:>12,.0f} {0.0:>6.1f}%")
for tau in (0.0, 0.34, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80):
    r = _simulate(use_signal=True, conf=conf, conf_threshold=tau, **base)
    pull = 100 * np.mean((conf >= tau) & (pred != 1))   # fraction of events we actually pull a side
    print(f"{('tau='+format(tau,'.2f')):>16} {r.net_pnl_eur:>11,.0f} {r.sharpe_net_ann:>8.2f} {r.n_bid_fills+r.n_ask_fills:>9,} {r.adverse_selection_eur:>12,.0f} {pull:>6.1f}%")
