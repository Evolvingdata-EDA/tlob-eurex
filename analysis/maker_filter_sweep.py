"""Test two microstructure entry filters on the h1 TLOB maker:
  (1) post-move cooldown: stand down for N events after any mid change;
  (2) wide-spread gate: don't quote when the book is > 1 tick wide.
Reports each vs the no-filter signal and the no-signal baseline.
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
spread_t = ((raw["ask1_price"] - raw["bid1_price"]) / spec.tick_size)
print(f"events with spread >1 tick: {100*(spread_t > 1.0001).mean():.2f}%")

base = dict(
    bid_px=raw["bid1_price"].to_numpy("float64"), ask_px=raw["ask1_price"].to_numpy("float64"),
    bid_qd=raw["bid1_qty"].to_numpy("float64"), ask_qd=raw["ask1_qty"].to_numpy("float64"),
    bid_exec=raw["bid_exec_qty"].to_numpy("float64"), ask_exec=raw["ask_exec_qty"].to_numpy("float64"),
    mid=raw["price"].to_numpy("float64"), pred=raw["Preds"].to_numpy("int64"), day_id=day_id,
    point_value=spec.point_value, maker_fee=0.0, taker_fee=spec.fee_per_side_native,
    tick=spec.tick_size, inv_cap=50, clip=1, cancel_advances_queue=True,
)

def row(name, **kw):
    r = _simulate(**{**base, **kw})
    f = r.n_bid_fills + r.n_ask_fills
    print(f"{name:>28} {r.net_pnl_eur:>11,.0f} {r.sharpe_net_ann:>7.2f} {f:>8,} {r.spread_earned_eur:>11,.0f} {r.adverse_selection_eur:>11,.0f}")

print(f"\n{'variant':>28} {'net EUR':>11} {'Sharpe':>7} {'fills':>8} {'spread':>11} {'adverse':>11}")
row("baseline (no signal)", use_signal=False)
row("signal (no filters)", use_signal=True)
for cd in (5, 10, 20, 50):
    row(f"signal + cooldown {cd}ev", use_signal=True, cooldown_events=cd)
row("signal + spread<=1t", use_signal=True, max_spread_ticks=1.0)
row("signal + cd10 + spread<=1t", use_signal=True, cooldown_events=10, max_spread_ticks=1.0)
