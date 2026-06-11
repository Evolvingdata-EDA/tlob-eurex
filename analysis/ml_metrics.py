"""Per-tick ML metrics (precision/recall/F1) for night checkpoints, from result.csv.

Robust evaluation per Leo's protocol: judge models on test-set ML metrics computed
over ALL test ticks (statistically meaningful), not on backtest PnL from ~30
path-dependent round-trips. Signals are gated like the strategy: up if
prob_0 > τ, down if prob_2 > τ (label convention 0=up, 1=hold, 2=down).

For each checkpoint and τ: number of signals, precision, the wrong-way rate
P(opposite label | signal) — the costly error — and recall/F1 per directional
class plus the directional macro-F1.

Usage:
    python analysis/ml_metrics.py [--taus 0.5 0.55 0.6 0.65]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CKPT_ROOT = Path(__file__).resolve().parents[1] / "checkpoints" / "TLOB"
OUT_CSV = Path(__file__).resolve().parents[1] / "results" / "ml_metrics_night_0611.csv"


def directional_metrics(df: pd.DataFrame, tau: float) -> dict:
    """Compute per-tick directional metrics at confidence gate `tau`.

    Args:
        df: result.csv frame with prob_0/prob_2 and integer `label`.
        tau: Confidence gate; tau<=0 means plain argmax over the 3 classes.

    Returns:
        Flat dict of counts, precision, wrong-way rate, recall, F1 per side
        plus directional macro-F1.
    """
    label = df["label"].to_numpy()
    p_up, p_dn = df["prob_0"].to_numpy(), df["prob_2"].to_numpy()
    if tau <= 0:  # argmax mode
        p_hold = df["prob_1"].to_numpy()
        pred = np.argmax(np.column_stack([p_up, p_hold, p_dn]), axis=1)
        sig_up, sig_dn = pred == 0, pred == 2
    else:
        sig_up, sig_dn = p_up > tau, p_dn > tau

    out: dict = {"tau": tau if tau > 0 else "argmax"}
    for name, sig, cls, opp in (("up", sig_up, 0, 2), ("down", sig_dn, 2, 0)):
        n_sig = int(sig.sum())
        n_cls = int((label == cls).sum())
        tp = int((sig & (label == cls)).sum())
        wrong = int((sig & (label == opp)).sum())
        prec = tp / n_sig if n_sig else np.nan
        rec = tp / n_cls if n_cls else np.nan
        f1 = 2 * prec * rec / (prec + rec) if n_sig and tp else 0.0
        out.update({
            f"n_{name}": n_sig,
            f"prec_{name}": round(prec, 4) if n_sig else np.nan,
            f"wrongway_{name}": round(wrong / n_sig, 4) if n_sig else np.nan,
            f"rec_{name}": round(rec, 4),
            f"f1_{name}": round(f1, 4),
        })
    f1s = [out["f1_up"], out["f1_down"]]
    out["macro_f1_dir"] = round(float(np.mean(f1s)), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.55, 0.6, 0.65])
    args = parser.parse_args()

    rows = []
    for ckpt in sorted(CKPT_ROOT.iterdir()):
        rc = ckpt / "result.csv"
        if not rc.exists() or "_SMK_" in ckpt.name:
            continue
        df = pd.read_csv(rc, usecols=["label", "prob_0", "prob_1", "prob_2"])
        run = ckpt.name.split("__")[-1]
        for tau in args.taus:
            rows.append({"run": run, "n_test": len(df), **directional_metrics(df, tau)})
    res = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res.to_string(index=False))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
