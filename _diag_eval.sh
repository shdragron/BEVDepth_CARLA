#!/usr/bin/env bash
# Diagnostic: evaluate the trained SEDAN checkpoint on another vehicle's val set.
# Tells us if that vehicle's data/eval pipeline is consistent (nonzero => OK).
# Usage: _diag_eval.sh <suv|bus> <gpu_id>
set -e
VEH=$1
GPU=$2
ENV=/NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs/bevdepth-b200
REPO=/home/hanyan_arch/viewpoint/BEVFormer/BEVDepth
cd "$REPO"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=$GPU
export TMPDIR=/tmp
export USE_WANDB=0
CKPT="$REPO/outputs/carla_sedan/BEVDepth-CARLA/7bex50bw/checkpoints/epoch=23-step=6528.ckpt"
LOG="$REPO/_diag_sedanckpt_on_${VEH}.log"
echo "DIAG START $(date)  sedan-ckpt -> ${VEH}-val (gpu$GPU)" | tee "$LOG"
"$ENV/bin/python" bevdepth/exps/nuscenes/carla/carla_${VEH}.py \
    --ckpt_path "$CKPT" -e --gpus 1 -b 8 2>&1 | tee -a "$LOG"
echo "DIAG DONE $(date) rc=${PIPESTATUS[0]}" | tee -a "$LOG"
