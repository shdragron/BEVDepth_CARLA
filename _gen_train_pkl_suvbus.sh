#!/usr/bin/env bash
# Generate suv + bus TRAIN info pkls (CPU/IO; safe alongside GPU training).
# Writes data/carla_infos_train_{suv,bus}.pkl only — does NOT touch BEVFormer pkls.
set -e
ENV=/NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs/bevdepth-b200
REPO=/home/hanyan_arch/viewpoint/BEVFormer/BEVDepth
cd "$REPO"
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO:$PYTHONPATH"
LOG="$REPO/_gen_train_pkl_suvbus.log"
echo "GEN PKL START $(date)" | tee "$LOG"
for veh in suv bus; do
  echo "=== [$veh] gen_info_carla --version v1.0-carla_${veh} --tag ${veh} ===" | tee -a "$LOG"
  "$ENV/bin/python" scripts/gen_info_carla.py \
      --version "v1.0-carla_${veh}" --tag "${veh}" 2>&1 | tee -a "$LOG"
done
echo "GEN PKL DONE $(date) rc=${PIPESTATUS[0]}" | tee -a "$LOG"
ls -la "$REPO"/data/carla_infos_train_*.pkl | tee -a "$LOG"
