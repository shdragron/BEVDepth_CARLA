#!/usr/bin/env bash
# Sedan CARLA training (CBGS OFF) for a persistent tmux session.
set -e
ENV=/NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs/bevdepth-b200
REPO=/home/hanyan_arch/viewpoint/BEVFormer/BEVDepth
cd "$REPO"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=BEVDepth-CARLA
export USE_CBGS=0                       # CBGS off (baseline regime, ~4x faster)
# file_system tensor sharing (set in carla_base.py) mmaps temp files under
# TMPDIR; pin it to /tmp (1.2 TB tmpfs, ~870 GB free) instead of the crowded
# 100 GB /dev/shm so dataloader workers aren't SIGKILLed (rc=137) under
# contention from the concurrent BEVDet train + VP eval.
export TMPDIR=/tmp
LOG="$REPO/_train_sedan_nocbgs.log"
echo "TMUX TRAIN START $(date)  (sedan, CBGS off, b16 x2GPU x accum2 = eff64, lr2e-4, GPU-safe)" | tee "$LOG"
"$ENV/bin/python" bevdepth/exps/nuscenes/carla/carla_sedan.py \
    --amp_backend native -b 16 --gpus 2 --accelerator gpu --strategy ddp \
    --accumulate_grad_batches 2 \
    --limit_val_batches 1.0 --check_val_every_n_epoch 1 2>&1 | tee -a "$LOG"
echo "TMUX TRAIN DONE $(date) rc=${PIPESTATUS[0]}" | tee -a "$LOG"
