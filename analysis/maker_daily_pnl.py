"""Per-day PnL of the h1 TLOB maker (signal, realistic queue), by calendar date,
to check whether losses land on known large-event days (Roll 4-5 Jun, NFP 5 Jun,
ECB 11 Jun)."""
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
EVENTS = {"20250604": "Roll", "20250605": "Roll/NFP", "20250611": "ECB"}

raw = pd.read_csv(D + "/result.csv").sort_values("timestamp").reset_index(drop=True)
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
spec = instruments.get_spec(SYM); sess = instruments.session_params(SYM)
raw = _filter_to_session(raw.set_index("timestamp"), sess.tz, sess.open_hour, sess.close_hour).reset_index()
loc = raw["timestamp"].dt.tz_localize("UTC").dt.tz_convert(sess.tz)
day_id = loc.dt.strftime("%Y%m%d").astype("int64").to_numpy()
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
day_pnl = sink["day_pnl"]

# daily realized-volatility proxy: intraday mid range in ticks + #1-tick moves
dfd = pd.DataFrame({"day": day_id, "mid": mid})
dmid = np.abs(np.diff(mid, prepend=mid[0])) / spec.tick_size
dfd["ticks"] = dmid
agg = dfd.groupby("day").agg(rng=("mid", lambda x: (x.max()-x.min())/spec.tick_size),
                             nticks=("ticks", lambda x: int((x >= 0.75).sum())))
rows = []
for d in sorted(day_pnl):
    ds = str(d)
    rows.append((f"{ds[:4]}-{ds[4:6]}-{ds[6:]}", day_pnl[d], agg.loc[d, "rng"], agg.loc[d, "nticks"], EVENTS.get(ds, "")))
tot = sum(day_pnl.values())
print(f"net {tot:,.0f} EUR over {len(rows)} days\n")
print(f"{'date':>12} {'PnL EUR':>10} {'range(t)':>9} {'#1t-moves':>10}  event")
for dt, p, rng, nt, ev in rows:
    star = " <<<" if p < -3000 else ""
    print(f"{dt:>12} {p:>10,.0f} {rng:>9.0f} {nt:>10,}  {ev}{star}")
neg = [(dt, p, ev) for dt, p, _, _, ev in rows if p < 0]
print(f"\nlosing days: {len(neg)}/{len(rows)}   sum of losing days: {sum(p for _,p,_ in neg):,.0f}")
ev_days = [(dt, p) for dt, p, _, _, ev in rows if ev]
print(f"event days PnL: " + ", ".join(f"{dt}={p:,.0f}" for dt, p in ev_days))
print(f"event days total: {sum(p for _,p in ev_days):,.0f}  ({100*sum(p for _,p in ev_days)/tot:.0f}% of net)")
