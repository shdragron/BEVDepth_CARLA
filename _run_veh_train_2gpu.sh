#!/usr/bin/env bash
# Per-vehicle CARLA training on 2 GPUs (DDP) — EXACT sedan config (BN batch 32).
# Usage: _run_veh_train_2gpu.sh <sedan|suv|bus>
# b16 x 2 GPU x accum2 = effective batch 64 -> lr 2e-4. CBGS off, TF32 off.
set -e
VEH=$1
ENV=/NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs/bevdepth-b200
REPO=/home/hanyan_arch/viewpoint/BEVFormer/BEVDepth
cd "$REPO"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=BEVDepth-CARLA
export USE_CBGS=0
export TMPDIR=/tmp
LOG="$REPO/_train_${VEH}_2gpu.log"
echo "TMUX TRAIN START $(date)  ($VEH, 2GPU DDP, b16 x2 x accum2 = eff64, lr2e-4, CBGS off, TF32 off)" | tee "$LOG"
"$ENV/bin/python" bevdepth/exps/nuscenes/carla/carla_${VEH}.py \
    --amp_backend native -b 16 --gpus 2 --accelerator gpu --strategy ddp \
    --accumulate_grad_batches 2 \
    --gradient_clip_val 35 \
    --limit_val_batches 1.0 --check_val_every_n_epoch 1 2>&1 | tee -a "$LOG"
echo "TMUX TRAIN DONE $(date) rc=${PIPESTATUS[0]}" | tee -a "$LOG"
