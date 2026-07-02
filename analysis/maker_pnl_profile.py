"""Risk-profile analysis of the h1 TLOB maker (signal, realistic queue): is the
negative PnL a continuous bleed or an event-driven left tail (picked off on large
moves / news)? Replays maker_sim._simulate with a series sink and characterizes
the per-event and per-day PnL distribution.
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

raw = pd.read_csv(D + "/result.csv")
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
raw = raw.sort_values("timestamp").reset_index(drop=True)
spec = instruments.get_spec(SYM); sess = instruments.session_params(SYM)
raw = _filter_to_session(raw.set_index("timestamp"), sess.tz, sess.open_hour, sess.close_hour).reset_index()
day_id = raw["timestamp"].dt.tz_localize("UTC").dt.tz_convert(sess.tz).dt.strftime("%Y%m%d").astype("int64").to_numpy()
mid = raw["price"].to_numpy("float64")

sink: dict = {}
_simulate(
    bid_px=raw["bid1_price"].to_numpy("float64"), ask_px=raw["ask1_price"].to_numpy("float64"),
    bid_qd=raw["bid1_qty"].to_numpy("float64"), ask_qd=raw["ask1_qty"].to_numpy("float64"),
    bid_exec=raw["bid_exec_qty"].to_numpy("float64"), ask_exec=raw["ask_exec_qty"].to_numpy("float64"),
    mid=mid, pred=raw["Preds"].to_numpy("int64"), day_id=day_id,
    point_value=spec.point_value, maker_fee=0.0, taker_fee=spec.fee_per_side_native, tick=spec.tick_size,
    use_signal=True, inv_cap=50, clip=1, cancel_advances_queue=True, series_sink=sink,
)
ev = sink["ev_delta"]; day_pnl = sink["day_pnl"]
net = ev.sum()
print(f"net (from ev sum): {net:,.0f} EUR   events={len(ev):,}")

# ---- ATTRIBUTE net PnL by the size of the per-step mid move ----
# ev_delta[i] books the equity change over step i = fills at i + inventory MTM
# over (mid_{i-1} -> mid_i). Bucketing ev_delta by |Δmid_i| PARTITIONS the net
# exactly, so it shows how much of the net comes from small-move steps (spread
# grind) vs large-move steps (getting run over). This is the real test.
dmid = np.abs(np.diff(mid, prepend=mid[0])) / spec.tick_size   # |Δmid_i| in ticks
edges = [0.25, 0.75, 1.25, 2.25, 1e9]
labels = ["~0 (<0.25t)", "0.5t", "1t", "1.5-2t", ">2t"]
bi = np.digitize(dmid, edges)
print(f"\n[net PnL attributed by per-step |Δmid|]  (sums to net)")
print(f"  {'bucket':>12} {'n_events':>10} {'net EUR':>12} {'% of net':>9} {'mean €/ev':>10}")
for b, lab in enumerate(labels):
    m = bi == b; s = ev[m].sum()
    print(f"  {lab:>12} {int(m.sum()):>10,} {s:>12,.0f} {100*s/net:>8.0f}% {ev[m].mean() if m.any() else 0:>10.3f}")
big = dmid >= 0.75
print(f"  --> steps with a move (|Δmid|>=1 tick): net {ev[big].sum():,.0f} EUR ({100*ev[big].sum()/net:.0f}% of net)")
print(f"      quiet steps (|Δmid|<1 tick):        net {ev[~big].sum():,.0f} EUR ({100*ev[~big].sum()/net:.0f}% of net)")

# ---- per-day distribution (news days) ----
dp = np.array(sorted(day_pnl.values()))
print(f"\n[per-day]  days={len(dp)}  negative days={int((dp<0).sum())} ({100*(dp<0).mean():.0f}%)")
print(f"  worst day {dp[0]:,.0f} | worst 5 days sum {dp[:5].sum():,.0f} = {100*dp[:5].sum()/net:.0f}% of net loss")
print(f"  median day {np.median(dp):,.0f}  mean day {dp.mean():,.0f}")
