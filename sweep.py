"""τ-sweep wrapper for tlob-eurex: registers Eurex instruments, then delegates
to tlob.backtest.rerun_backtest (which needs a populated registry to resolve
fees/sessions for FBTP/FOAT/FGB*).

Usage (same CLI as tlob.backtest.rerun_backtest):
    python sweep.py <ckpt_dir> [--threshold 0.7] [...]
"""
import runpy

import instruments_eurex  # noqa: F401  (side-effect: populates tlob.instruments)

runpy.run_module("tlob.backtest.rerun_backtest", run_name="__main__")
