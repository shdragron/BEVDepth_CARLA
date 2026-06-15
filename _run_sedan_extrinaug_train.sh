#!/usr/bin/env bash
# BEVDepth CARLA sedan + train-time EXTRINSIC-rotation aug (±20°, p=0.5, per-cam).
# EXACT sedan recipe (_run_sedan_train.sh): b16 x2GPU x accum2=eff64, lr2e-4,
# CBGS off, TF32 off, NO grad-clip. Only diff = carla_sedan_extrinaug config.
set -e
ENV=/NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs/bevdepth-b200
REPO=/home/hanyan_arch/viewpoint/BEVFormer/BEVDepth
cd "$REPO"
export CUDA_HOME="$ENV"; export PATH="$ENV/bin:$PATH"; export PYTHONPATH="$REPO:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=BEVDepth-CARLA
export USE_CBGS=0
export TMPDIR=/tmp
LOG="$REPO/_train_sedan_extrinaug.log"
echo "TRAIN START $(date)  (sedan+extrinaug ±20deg p0.5, 2GPU DDP eff64 lr2e-4 CBGS off)" | tee "$LOG"
"$ENV/bin/python" bevdepth/exps/nuscenes/carla/carla_sedan_extrinaug.py \
    --amp_backend native -b 16 --gpus 2 --accelerator gpu --strategy ddp \
    --accumulate_grad_batches 2 \
    --limit_val_batches 1.0 --check_val_every_n_epoch 1 2>&1 | tee -a "$LOG"
echo "TRAIN DONE $(date) rc=${PIPESTATUS[0]}" | tee -a "$LOG"
