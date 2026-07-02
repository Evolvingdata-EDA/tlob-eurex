#!/usr/bin/env bash
# Three 1s-sign (mm_h1_sign) TLOB runs: predict Bund/OAT/BTP with FESX+CL added
# as input via OUTER join. Replicates the FGBL_ZN_outer template exactly, only
# swapping data_sources + run_tag. Common window auto-trims to 2025-07-01..2026-05-29.
set -euo pipefail
cd /home/rig/PythonProjects/tlob-eurex
PY=/home/rig/PythonProjects/tlob-a2a/.venv/bin/python

# --- farm symlinks for the two new sources (idempotent) ---
[ -e data/FESX_TBBO ] || ln -s FESX.v.0 data/FESX_TBBO
[ -e data/CL_TBBO ]   || ln -s CL.n.0   data/CL_TBBO
echo "FESX_TBBO days: $(ls data/FESX_TBBO/ | wc -l)   CL_TBBO days: $(ls data/CL_TBBO/ | wc -l)"

# --- pick the freest GPU (lowest util, then lowest mem) ---
GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
      | sort -t, -k2 -n -k3 -n | head -1 | cut -d, -f1 | tr -d ' ')
echo "Using GPU $GPU"
export CUDA_VISIBLE_DEVICES=$GPU

COMMON=(+model=tlob experiment.fee=0.01 dataset.is_databento=True
  experiment.filter_session=True dataset.batch_size=1024 experiment.is_wandb=False
  experiment.max_epochs=20 dataset.n_lob_levels=1
  dataset.labeling_type=SMOOTHED_MID_TIME experiment.horizon=1
  dataset.len_trade_window=10 dataset.label_threshold_ticks=0.0
  experiment.is_3d=False dataset.join_type=OUTER
  experiment.backtest_frictionless=True experiment.is_data_preprocessed=False
  hydra.job.chdir=False)

run() {  # $1=anchor  $2=tag
  echo "=== RUN $2 : [$1,FESX_TBBO,CL_TBBO] ==="
  $PY main.py "${COMMON[@]}" \
    "dataset.data_sources=[${1},FESX_TBBO,CL_TBBO]" \
    "experiment.run_tag=$2"
}

run FGBL_TBBO mm_h1_sign_FGBL_FESX_CL_outer
run FOAT_TBBO mm_h1_sign_FOAT_FESX_CL_outer
run FBTP_TBBO mm_h1_sign_FBTP_FESX_CL_outer
echo "ALL THREE DONE"
