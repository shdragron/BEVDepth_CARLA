#!/usr/bin/env bash
# Per-vehicle CARLA training (CBGS off) on ONE GPU, for a persistent tmux session.
# Usage: _run_veh_train.sh <suv|bus> <gpu_id>
# b16 x 1 GPU x accum4 = effective batch 64 -> lr 2e-4 (same regime as sedan).
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
export WANDB_PROJECT=BEVDepth-CARLA
export USE_CBGS=0          # CBGS off (baseline regime)
export TMPDIR=/tmp         # file_system tensor sharing -> big tmpfs, not /dev/shm
LOG="$REPO/_train_${VEH}_nocbgs.log"
echo "TMUX TRAIN START $(date)  ($VEH, CBGS off, b16 x1GPU(gpu$GPU) x accum4 = eff64, lr2e-4, TF32-off)" | tee "$LOG"
"$ENV/bin/python" bevdepth/exps/nuscenes/carla/carla_${VEH}.py \
    --amp_backend native -b 16 --gpus 1 --accelerator gpu \
    --accumulate_grad_batches 4 \
    --limit_val_batches 1.0 --check_val_every_n_epoch 1 2>&1 | tee -a "$LOG"
echo "TMUX TRAIN DONE $(date) rc=${PIPESTATUS[0]}" | tee -a "$LOG"
