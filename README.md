# BEVDepth-CARLA

Camera-only 3D object detection on the **CARLA "geobev" per-vehicle (viewpoint)
dataset**, adapted from [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
(Megvii-BaseDetection). This fork trains and evaluates a single-frame BEVDepth
detector on three vehicle platforms — **sedan / suv / bus** — that differ only by
camera mount height, using the 6 CARLA classes and the nuScenes-format eval
protocol so the numbers are directly comparable to the BEVFormer CARLA baseline.

> **Attribution / License.** This is a derivative work of BEVDepth, released under
> the **MIT License (Copyright © 2022 Megvii-BaseDetection)** — see
> [`LICENSE.md`](LICENSE.md), which is unchanged. All upstream BEVDepth / BEVStereo
> / MatrixVT code and credits remain with the original authors; this fork only adds
> the CARLA dataset, exp configs, evaluator changes, and the training/eval glue
> documented below. Please cite the original BEVDepth paper
> ([arXiv:2206.10092](https://arxiv.org/abs/2206.10092)) when using this work.

---

## What this fork adds

* **CARLA dataset** (`bevdepth/datasets/carla_det_dataset.py`): nuScenes-format
  per-vehicle DBs, `.npz` lidar loader, and **dense DPT depth** as depth-loss GT
  (instead of sparse lidar projection).
* **6 classes**: `car, truck, bus, motorcycle, bicycle, pedestrian`.
* **Visibility-based GT** (`gt_visibility_min=2`): GT boxes are kept by
  `visibility_token >= 2` (matching the BEVFormer baseline / the dataset's
  `valid_flag`), **not** by `num_lidar_pts > 0`. This applies to **both training
  and evaluation** so the two are consistent.
* **6-class NDS evaluator** (`bevdepth/evaluators/det_evaluators.py`): recomputes
  mAP/NDS over exactly the 6 CARLA classes (the devkit's 10-class average is
  diluted by 4 absent classes), runs the stock nuScenes devkit on the custom
  `v1.0-carla_*_eval` DBs, and prints a scrapeable
  `[CARLA-EVAL] 6-class mAP=.. NDS=..` line.
* **CBGS** (class-balanced group sampling) on by default for CARLA, to lift the
  rare-class AP (truck / motorcycle / bicycle).
* **Per-epoch 6-class validation** logged to **Weights & Biases** (`val/NDS`,
  `val/mAP`).
* Exp configs `bevdepth/exps/nuscenes/carla/carla_{sedan,suv,bus}.py`.

---

## 1. Environment

Validated on **NVIDIA B200 (Blackwell, sm_100) / CUDA 12.8**, Python 3.10, with:

| package | version |
|---|---|
| torch / torchvision | 2.x (built for your CUDA; cu128 for B200). Validated on **2.11.0+cu128 / cuDNN 9.19** |
| pytorch-lightning | **1.6.2** |
| torchmetrics | **0.7.2** |
| mmcv-full | **1.7.1** (build from source, `-std=c++17`) |
| mmdet | **2.14.0** (raise its mmcv cap) |
| mmdet3d | **0.17.1** (+ `patches/mmdet3d-0.17.1-torch2x.patch`) |
| numba, nuscenes-devkit, tensorboardX, wandb | latest |

> **Why these three need a touch on torch 2.x.** The mm-stack pinned above is the
> 2021 line that BEVDepth/BEVFormer were written against; it predates torch 2.x and
> needs three *small, verified* adjustments to build/import on a modern (B200 /
> torch 2.x) box. None change any op math — they are header/flag/version fixes only,
> so results are identical to the original stack. (For older GPUs you can instead
> follow upstream BEVDepth: torch 1.9.0 + mmdet3d v1.0.0rc4, no patches.)
>
> 1. **mmcv-full 1.7.1** pins `-std=c++14`, which fails against torch ≥ 2.x headers
>    (`#error You need C++17 ...`). Build it from source with `-std=c++17`.
> 2. **mmdet 2.14.0** caps `mmcv_maximum_version='1.4.0'`, rejecting 1.7.1 — raise
>    the cap (one line; mmdet 2.x ships no CUDA ops, so it is otherwise pure-Python).
> 3. **mmdet3d 0.17.1** caps mmcv the same way *and* its 6 point-cloud ops include
>    `THC/THC.h`, removed in torch ≥ 1.11. Apply the bundled
>    [`patches/mmdet3d-0.17.1-torch2x.patch`](patches/mmdet3d-0.17.1-torch2x.patch)
>    (9 KB: THC→ATen headers, `-std=c++17`, mmcv cap, a couple of `np.long` fixes),
>    then build. *Note:* BEVDepth is camera-only and uses only pure-Python mmdet3d
>    symbols (`build_neck`, `CenterHead`/`circle_nms`, `draw_heatmap_gaussian`/
>    `gaussian_radius`, `clip_sigmoid`) — none of the THC point ops — so the patch
>    is only needed to make `pip install -e .` succeed, not for inference.

```bash
# 0) env with a CUDA-matched torch (cu128 for B200).
conda create -n bevdepth python=3.10 -y && conda activate bevdepth
#    ... install torch/torchvision for your CUDA here ...

# 1) mmcv-full 1.7.1 from source with C++17 (the only change vs stock):
git clone -b v1.7.1 https://github.com/open-mmlab/mmcv.git && cd mmcv
sed -i -E "s/-std=c\+\+14/-std=c++17/g" setup.py
MMCV_WITH_OPS=1 CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="10.0" \
    pip install -e . && cd ..

# 2) mmdet 2.14.0 + raise its mmcv cap so 1.7.1 is accepted:
pip install mmdet==2.14.0
sed -i -E "s/mmcv_maximum_version = '[0-9.]+'/mmcv_maximum_version = '1.99.0'/" \
    "$(python -c 'import mmdet,os;print(os.path.dirname(mmdet.__file__))')/__init__.py"

# 3) mmdet3d 0.17.1 (stock) + the bundled torch-2.x patch, then build:
git clone -b v0.17.1 https://github.com/open-mmlab/mmdetection3d.git && cd mmdetection3d
git apply /path/to/BEVDepth_CARLA/patches/mmdet3d-0.17.1-torch2x.patch
CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="10.0" pip install -e . && cd ..

# 4) PyTorch-Lightning 1.6.2 (BEVDepth uses the PL-1.x Trainer API).
#    PL 1.6.2 ships invalid metadata that pip>=24.1 rejects, so pin pip first:
pip install "pip<24.1"
pip install --no-deps pytorch_lightning==1.6.2 torchmetrics==0.7.2 \
            tensorboardX pyDeprecate==0.3.2
pip install numba nuscenes-devkit wandb

# 5) compile this repo's voxel-pooling CUDA ops (CUDA_HOME = your torch's CUDA):
CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="10.0" \
    python setup.py develop          # or: build_ext --inplace
```

Run the exps **from the repo root** (so `import bevdepth` and `data/carla`
resolve). Log in to wandb once (`wandb login`) or export `WANDB_API_KEY`.

---

## 2. Data preparation

The CARLA geobev dataset is a set of **per-vehicle nuScenes-format DBs** under one
root (`carla_geobev/`): train DBs `v1.0-carla_{sedan,suv,bus}` (220 scenes each)
and eval DBs `v1.0-carla_{sedan,suv,bus}_eval` (48 val scenes, from
`split/val.txt`). Images / lidar / DPT-depth live under the same root.

```bash
# symlink the dataset root to ./data/carla (images, lidar, DPT depth, DBs)
ln -s /path/to/carla_geobev data/carla

# build the info pkls -> ./data/carla_infos_{train,val}_<veh>.pkl
# train (per vehicle):
python scripts/gen_info_carla.py      --version v1.0-carla_sedan      --tag sedan
python scripts/gen_info_carla.py      --version v1.0-carla_suv        --tag suv
python scripts/gen_info_carla.py      --version v1.0-carla_bus        --tag bus
# eval / val (per vehicle):
python scripts/gen_info_carla_eval.py --version v1.0-carla_sedan_eval --tag sedan
python scripts/gen_info_carla_eval.py --version v1.0-carla_suv_eval   --tag suv
python scripts/gen_info_carla_eval.py --version v1.0-carla_bus_eval   --tag bus
```

Each `gen_info_carla_eval` run prints `val samples: 3792` and drops the one DB
scene not in `val.txt` (`scene_0260`) so the val set matches the baseline exactly.

---

## 3. Training

One exp per vehicle; only `VEHICLE` differs. CBGS is on, validation runs the
**6-class** eval **every epoch** and logs `val/NDS` / `val/mAP` to wandb.

```bash
# sedan (use carla_suv.py / carla_bus.py for the others)
CUDA_VISIBLE_DEVICES=0 python bevdepth/exps/nuscenes/carla/carla_sedan.py \
    --amp_backend native -b 32 --gpus 1 \
    --limit_val_batches 1.0 --check_val_every_n_epoch 1
```

* `-b` = batch size **per device**; learning rate scales automatically
  (`2e-4/64 * b * gpus`). `-b 64 --gpus 1` reproduces the original setup if it
  fits your GPU; `-b 32` is a lighter alternative.
* `--limit_val_batches 1.0 --check_val_every_n_epoch 1` enables the full 6-class
  validation every epoch (the default disables validation).
* fp32, no EMA (set in `carla_*.py` for a fair comparison with BEVFormer).
* Disable wandb with `USE_WANDB=0`; set the project with `WANDB_PROJECT=...`.
* Multi-GPU: `--gpus 2 --strategy ddp` (lr scales with total batch).

Checkpoints land in `./outputs/carla_<veh>/lightning_logs/.../checkpoints/`.

---

## 4. Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python bevdepth/exps/nuscenes/carla/carla_sedan.py \
    --ckpt_path /path/to/checkpoint.ckpt -e --gpus 1 -b 8
```

Output (the `[CARLA-EVAL]` line is the comparable metric):

```
[CARLA-EVAL] version=v1.0-carla_sedan_eval eval_split=val min_visibility>=2 pred_samples=3792 carla_scenes=48
[CARLA-EVAL] 6-class mAP=0.2955 NDS=0.3278
[CARLA-METRICS-JSON] {...full per-class AP / 6-class & 10-class TP errors / NDS...}
```

* The evaluator loads GT from the per-vehicle `v1.0-carla_<veh>_eval` DB, keeps
  `visibility_token >= 2`, and computes mAP/NDS over exactly the 6 CARLA classes.
* `NDS` / `mAP` are the 6-class scores; `*_allclass` / `*_10class` keys in the
  JSON are the diluted 10-class devkit values (for reference only).

---

## 5. Notes / known characteristics

* **Single-frame velocity.** These exps are single-frame (`num_sweeps=1`,
  `key_idxes=[]`), so velocity is unestimable: `mAVE ≈ object speed` and the
  velocity term contributes ~0 to NDS (caps NDS near ~0.40). Enable multi-frame
  to recover it.
* **CBGS** chiefly helps the rare classes (truck / motorcycle / bicycle) and
  hence the 6-class mAP/NDS; it only affects `train_dataloader` (val/eval use
  `use_cbgs=False`).
* **DPT depth** is used as depth-loss GT (dense), only when `return_depth=True`
  (training); evaluation never loads depth, so it is image-only.
* **TF32 is disabled (full fp32).** `carla_base.py` sets
  `cuda.matmul.allow_tf32 = cudnn.allow_tf32 = False` +
  `set_float32_matmul_precision('highest')`. Even at Trainer `precision=32`, TF32
  stays on for cuDNN convs by default, and on B200 the depth net's TF32 conv path
  is **cuDNN-version dependent** — cuDNN 9.19 is stable, but e.g. cuDNN 9.7.1 picks
  an unstable TF32 conv kernel and `depth_loss` diverges (climbs without bound)
  while `detection_loss` stays flat. Forcing full fp32 makes training stable and
  bit-reproducible across cuDNN/GPU versions; it is also what the fair-comparison
  `precision=32` intends. (Cost: a little slower; B200 fp32 throughput makes it
  negligible.)

---

## Acknowledgements

Built on [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) (and
BEVStereo / MatrixVT) by Megvii-BaseDetection. See [`LICENSE.md`](LICENSE.md) (MIT).
