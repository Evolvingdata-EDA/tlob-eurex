#!/bin/bash
# FGBL (Bund) microstructural-feature experiment: add_micro + add_micro_extra
# (Tushar-baseline signals) on two targets, sequential on GPU1.
#   Run A: 1s directional  — SMOOTHED_MID_TIME, horizon=1 (seconds)
#   Run B: 5-tick           — SMOOTHED_MID,      horizon=5 (events)
set -euo pipefail
cd /home/rig/PythonProjects/tlob-eurex
PY=/home/rig/PythonProjects/tlob-a2a/.venv/bin/python

COMMON=(
  +model=tlob
  experiment.fee=0.01
  dataset.is_databento=True
  experiment.filter_session=True
  dataset.batch_size=1024
  experiment.is_wandb=False
  experiment.max_epochs=20
  dataset.data_sources=[FGBL_TBBO]
  dataset.n_lob_levels=1
  dataset.len_trade_window=10
  dataset.add_micro=True
  dataset.add_micro_extra=True
  experiment.is_data_preprocessed=False
  experiment.gpu_id=1
  hydra.job.chdir=False
)

echo "===== RUN A: FGBL 1s SMOOTHED_MID_TIME (h=1s) ====="
"$PY" main.py "${COMMON[@]}" \
  dataset.labeling_type=SMOOTHED_MID_TIME experiment.horizon=1 \
  experiment.run_tag=fgbl_1s_microx

echo "===== RUN B: FGBL 5-tick SMOOTHED_MID (h=5 events) ====="
"$PY" main.py "${COMMON[@]}" \
  dataset.labeling_type=SMOOTHED_MID experiment.horizon=5 \
  experiment.run_tag=fgbl_5tick_microx

echo "===== ALL RUNS DONE ====="
