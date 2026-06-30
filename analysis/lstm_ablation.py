"""Incremental-source ablation: does adding cross-asset inputs help forecast the Bund?

A standard LSTM predicts the *sign of the next-bar return of FGBL (Bund)* from a
window of past bar returns. We start from the bond's own history (autoregressive
baseline) and greedily add the macro/energy/risk source that most improves
validation accuracy, one at a time, recording the test-accuracy curve. Run for
each bar frequency (1min/15min/1h/1d) so the *horizon-dependence* of cross-asset
value is the headline result.

Crucially the target return over (t-1, t] is predicted from features strictly up
to t-1, so the model can only exploit *leading* (lag>=1) information — consistent
with analysis/bar_lead_lag.py. We therefore expect ~no benefit intraday and the
best lift at the daily horizon (oil/gas reversal channel).

Outputs (results/lstm_ablation/):
    curve_{freq}.csv     — greedy step, source added, val/test acc (mean+/-std over seeds)
    ablation_curve.png   — test acc vs n_sources, one line per horizon
    summary.md

Usage:
    python analysis/lstm_ablation.py [--seeds 3] [--freqs 1min,15min,1h,1d]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bar_lead_lag import DRIVERS, return_panel  # reuse aligned return panels

TARGET = "FGBL"  # Bund — outright
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "lstm_ablation"

# freq -> (lookback window in bars, batch size, max epochs)
FREQ_CFG: dict[str, tuple[int, int, int]] = {
    "1min": (60, 4096, 25),
    "15min": (32, 1024, 40),
    "1h": (24, 256, 60),
    "1d": (20, 64, 80),
}
HIDDEN = 64
LR = 1e-3
PATIENCE = 6


def pick_device() -> str:
    """Free GPU (<50% util and <50% mem) else CPU — respects co-located training."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    except Exception:
        return "cpu"
    for line in out:
        idx, util, used, total = (x.strip() for x in line.split(","))
        if float(util) < 50 and float(used) / float(total) < 0.5:
            return f"cuda:{idx}"
    return "cpu"


def make_windows(r: pd.DataFrame, sources: list[str], L: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows of source returns -> next-bar Bund direction sign.

    X[i] = returns of `sources` over rows [i, i+L)  (info up to t-1)
    y[i] = sign of FGBL return at row i+L in {-1, 0, +1}  (target move into t)

    Flat (y==0) bars are kept here and masked out per-split in train_eval, so we
    only score *direction* on bars that actually moved (binary up/down).
    """
    feats = r[sources].values.astype(np.float32)          # [N, F]
    tgt = np.sign(r[TARGET].values).astype(np.int64)       # [N] in {-1,0,1}
    n = len(r) - L
    X = np.stack([feats[i:i + L] for i in range(n)])       # [n, L, F]
    y = tgt[L:L + n]                                        # [n]
    return X, y


def balanced_acc(pred: np.ndarray, y: np.ndarray) -> float:
    """Mean of per-class recall for binary {0=down,1=up} — chance = 0.5 regardless of base rate."""
    recalls = [(pred[y == c] == c).mean() for c in (0, 1) if (y == c).any()]
    return float(np.mean(recalls)) if recalls else 0.0


class LSTMClf(nn.Module):
    """Minimal single-layer LSTM classifier. Input [B, L, F] -> logits [B, 2]."""

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_feat, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, F]
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])  # [B, 2]


def train_eval(r: pd.DataFrame, sources: list[str], L: int, batch: int,
               max_epochs: int, seed: int, device: str) -> tuple[float, float]:
    """Train on a temporal 70/15/15 split; return (val_acc, test_acc) at best val epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, ysign = make_windows(r, sources, L)
    n = len(X)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)

    mu, sd = X[:i_tr].mean((0, 1)), X[:i_tr].std((0, 1)) + 1e-8  # fit on train only
    X = (X - mu) / sd

    def loader(a: int, b: int, shuffle: bool) -> DataLoader:
        # keep only bars that moved; binary label up(1)/down(0)
        keep = ysign[a:b] != 0
        xs = torch.from_numpy(X[a:b][keep])
        ys = torch.from_numpy((ysign[a:b][keep] > 0).astype(np.int64))
        return DataLoader(TensorDataset(xs, ys), batch_size=batch, shuffle=shuffle, drop_last=False)

    tr, va, te = loader(0, i_tr, True), loader(i_tr, i_va, False), loader(i_va, n, False)
    model = LSTMClf(len(sources)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()

    @torch.no_grad()
    def acc(dl: DataLoader) -> float:
        model.eval()
        preds, ys = [], []
        for xb, yb in dl:
            preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
            ys.append(yb.numpy())
        return balanced_acc(np.concatenate(preds), np.concatenate(ys))

    best_va, best_te, bad = 0.0, 0.0, 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            lossf(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        va_acc = acc(va)
        if va_acc > best_va:
            best_va, best_te, bad = va_acc, acc(te), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_va, best_te


def greedy(freq: str, seeds: int, device: str) -> pd.DataFrame:
    """Greedy forward selection of sources for one frequency. Returns the curve."""
    L, batch, max_epochs = FREQ_CFG[freq]
    r = return_panel(freq)
    selected = [TARGET]  # autoregressive baseline: Bund on its own past
    remaining = list(DRIVERS)
    rows = []

    def evaluate(srcs: list[str]) -> tuple[float, float, float]:
        vs = [train_eval(r, srcs, L, batch, max_epochs, s, device) for s in range(seeds)]
        va = np.mean([v for v, _ in vs])
        te = np.array([t for _, t in vs])
        return va, te.mean(), te.std()

    va, te_m, te_s = evaluate(selected)
    rows.append({"step": 0, "added": "FGBL(self)", "n_sources": 1,
                 "set": "+".join(selected), "val_acc": va, "test_acc": te_m, "test_std": te_s})
    print(f"[{freq}] base FGBL(self): val={va:.4f} test={te_m:.4f}±{te_s:.4f}")

    step = 1
    while remaining:
        scored = [(s, *evaluate(selected + [s])) for s in remaining]
        s_best, va_b, te_b, ts_b = max(scored, key=lambda x: x[1])  # by val acc
        selected.append(s_best)
        remaining.remove(s_best)
        rows.append({"step": step, "added": s_best, "n_sources": len(selected),
                     "set": "+".join(selected), "val_acc": va_b, "test_acc": te_b, "test_std": ts_b})
        print(f"[{freq}] +{s_best:4s}: val={va_b:.4f} test={te_b:.4f}±{ts_b:.4f}  set={'+'.join(selected)}")
        step += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--freqs", default="1min,15min,1h,1d")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device={device}  seeds={args.seeds}")

    curves = {}
    for freq in args.freqs.split(","):
        df = greedy(freq, args.seeds, device)
        df.to_csv(OUT_DIR / f"curve_{freq}.csv", index=False)
        curves[freq] = df

    fig, ax = plt.subplots(figsize=(8, 5))
    for freq, df in curves.items():
        ax.errorbar(df["n_sources"], df["test_acc"], yerr=df["test_std"],
                    marker="o", capsize=3, label=freq)
    ax.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
    ax.set_xlabel("# input sources (greedy order)")
    ax.set_ylabel("test balanced accuracy (Bund next-bar direction, non-flat)")
    ax.set_title("Incremental-source ablation — does adding sources help forecast the Bund?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_curve.png", dpi=140)
    plt.close(fig)

    lines = ["# Incremental-source LSTM ablation — Bund (FGBL) next-bar direction\n",
             f"Seeds={args.seeds}. Greedy forward selection; test acc = mean±std over seeds.\n"]
    for freq, df in curves.items():
        base, best = df.iloc[0]["test_acc"], df["test_acc"].max()
        lines.append(f"\n## {freq}: baseline {base:.4f} -> best {best:.4f} (+{best-base:+.4f})\n")
        lines.append("| step | added | n_src | val_acc | test_acc | test_std |")
        lines.append("|---|---|---|---|---|---|")
        for _, x in df.iterrows():
            lines.append(f"| {x.step} | {x.added} | {x.n_sources} | {x.val_acc:.4f} | {x.test_acc:.4f} | {x.test_std:.4f} |")
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"Wrote {OUT_DIR}/summary.md")


if __name__ == "__main__":
    main()
