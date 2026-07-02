#!/bin/bash
# One FGBL microx run. Args: <gpu_id> <labeling_type> <horizon> <run_tag>
set -euo pipefail
cd /home/rig/PythonProjects/tlob-eurex
PY=/home/rig/PythonProjects/tlob-a2a/.venv/bin/python
GPU="$1"; LABEL="$2"; HORIZON="$3"; TAG="$4"
"$PY" main.py \
  +model=tlob experiment.fee=0.01 dataset.is_databento=True experiment.filter_session=True \
  dataset.batch_size=1024 experiment.is_wandb=False experiment.max_epochs=20 \
  dataset.data_sources=[FGBL_TBBO] dataset.n_lob_levels=1 dataset.len_trade_window=10 \
  dataset.add_micro=True dataset.add_micro_extra=True \
  dataset.label_threshold_ticks=0.0 \
  experiment.is_data_preprocessed=False experiment.gpu_id="$GPU" hydra.job.chdir=False \
  dataset.labeling_type="$LABEL" experiment.horizon="$HORIZON" experiment.run_tag="$TAG"
echo "===== DONE $TAG ====="
