# Copyright (c) Megvii Inc. All rights reserved.
"""Single-frame BEVDepth on the CARLA per-vehicle (viewpoint) dataset.

Shared base for the sedan / suv / bus CARLA experiments. Each subclass only sets
``VEHICLE``; everything else (class set, detection head, GT filter, data paths,
evaluator DB) is derived from it.

Matches the BEVFormer CARLA setup so the two are comparable:
  * 6 CARLA classes
  * visibility_token >= 2 GT filter  (via CarlaDetDataset, NOT num_lidar_pts)
  * per-vehicle DB:  train = v1.0-carla_<veh>,  eval = v1.0-carla_<veh>_eval
  * .npz lidar (depth supervision) handled by CarlaDetDataset
  * info pkls produced by scripts/gen_info_carla{,_eval}.py using the shared
    split/{train,val}.txt (train 220 scenes / val 48 scenes)
"""
import copy
import os

import torch
import torch.multiprocessing as _mp

from bevdepth.datasets.carla_det_dataset import CarlaDetDataset
from bevdepth.exps.nuscenes.base_exp import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
from bevdepth.exps.nuscenes.base_exp import head_conf

# --- DataLoader tensor sharing: file_system, not file_descriptor (/dev/shm) ---
# On this shared B200 box /dev/shm is a 100 GB container mount that other jobs
# (the concurrent BEVDet train + VP eval) keep ~2/3 full, leaving < 35 GB. The
# default 'file_descriptor' strategy stages every prefetched batch through
# /dev/shm; with num_workers=4 x 2 DDP ranks x prefetch=2 that is ~16 batches in
# flight, which overruns the free shm and the OS SIGKILLs a worker on the very
# first fetch ("DataLoader worker killed by signal: Killed" -> rc=137). The
# 'file_system' strategy instead mmaps temp files under $TMPDIR (=/tmp, a 1.2 TB
# tmpfs with ~870 GB free here), so we get 25x the headroom and never touch the
# crowded /dev/shm. (FD ulimit is 1 M, so file_system's per-tensor fd cost is a
# non-issue.) The run script pins TMPDIR=/tmp to keep this on the big tmpfs.
try:
    _mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass

# --- Force full-precision fp32 (no TF32) for stability across cuDNN versions ---
# These exps run Trainer precision=32, but TF32 (10-bit mantissa) silently stays
# on for cuDNN convolutions (cudnn.allow_tf32 defaults True). On B200 the depth
# net's TF32 conv path is *cuDNN-version dependent*: cuDNN 9.19 is stable, but
# cuDNN 9.7.1 picks a TF32 conv kernel that is numerically unstable here -> the
# depth distribution's gradient drifts every step and `depth_loss` climbs without
# bound (e.g. 19 -> 47 -> 107 ...) while detection_loss stays flat (it tolerates
# the noise). Verified: disabling TF32 holds depth_loss flat (~20) to step 119 on
# the env that otherwise diverged. Turning TF32 off makes precision=32 actually
# full fp32 (which is what the fair-comparison setup intends) and removes the
# hardware/cuDNN-version dependence, so a fresh clone trains identically anywhere.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision('highest')
except AttributeError:
    pass

# Enable wandb logging by default for these exps (override with USE_WANDB=0).
os.environ.setdefault('USE_WANDB', '1')

# nuScenes-format CARLA DB (tables + GT). The evaluator loads GT from here (the
# DB root), not the image symlink data_root, to compute mAP/NDS.
CARLA_DB_ROOT = '/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev'

CARLA_CLASSES = [
    'car',
    'truck',
    'bus',
    'motorcycle',
    'bicycle',
    'pedestrian',
]

# CenterHead task grouping for the 6 CARLA classes (5 task heads).
CARLA_TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=1, class_names=['truck']),
    dict(num_class=1, class_names=['bus']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=1, class_names=['pedestrian']),
]

carla_head_conf = copy.deepcopy(head_conf)
carla_head_conf['tasks'] = CARLA_TASKS
# One NMS min_radius per task (5 tasks).
carla_head_conf['test_cfg']['min_radius'] = [4, 12, 10, 1, 0.175]


class CarlaBEVDepthBase(BaseBEVDepthLightningModel):
    """Base CARLA exp; subclasses set ``VEHICLE`` in {'sedan','suv','bus'}."""

    VEHICLE = 'sedan'

    def __init__(self, **kwargs):
        super().__init__(class_names=CARLA_CLASSES,
                         head_conf=carla_head_conf,
                         **kwargs)
        veh = self.VEHICLE
        # data_root holds the image/lidar tree (symlink -> carla_geobev);
        # info pkls live alongside it under ./data.
        self.data_root = 'data/carla'
        self.train_info_paths = f'data/carla_infos_train_{veh}.pkl'
        self.val_info_paths = f'data/carla_infos_val_{veh}.pkl'
        self.predict_info_paths = f'data/carla_infos_val_{veh}.pkl'
        # CTS/VP condition-pkl injection (null-effect when unset): the
        # bev_det_benchmark runners point `-e` eval at a swapped condition pkl
        # via CARLA_VAL_INFO so the model runs unchanged. Only val/predict are
        # read in eval; training (CARLA_VAL_INFO unset) is unaffected.
        _cond_pkl = os.environ.get('CARLA_VAL_INFO')
        if _cond_pkl:
            self.val_info_paths = _cond_pkl
            self.predict_info_paths = _cond_pkl
        # Evaluator reads GT from the per-vehicle eval DB.
        self.evaluator.data_root = CARLA_DB_ROOT
        self.evaluator.version = f'v1.0-carla_{veh}_eval'
        # NDS/mAP are recomputed over exactly these 6 classes inside
        # DetNuscEvaluator._evaluate_single (over self.class_names), since the
        # devkit DetectionConfig hard-requires all 10 classes. No config change
        # needed here; the evaluator already has the 6 CARLA class_names.
        # visibility >= 2 GT filter (matches BEVFormer); CarlaDetDataset applies
        # it via _keep_ann, replacing the default num_lidar_pts+radar>0 filter.
        self.gt_visibility_min = 2
        # CBGS (class-balanced group sampling): resample the train set so rare
        # classes (truck/motorcycle/bicycle) appear more often -> higher per-class
        # AP -> higher 6-class mAP/NDS. Standard in BEVDepth's reported numbers;
        # only affects train_dataloader (val/eval pass use_cbgs=False). On by
        # default; set USE_CBGS=0 to disable (it ~4x's the per-epoch samples).
        self.data_use_cbgs = os.environ.get('USE_CBGS', '1') == '1'
        # Use the CARLA dataset (npz lidar loader + visibility filter).
        self.dataset_class = CarlaDetDataset

        # depth_net's DeformConv2dPack ships im2col_step=128; mmcv asserts
        # input.size(0) % min(im2col_step, input.size(0)) == 0. The deform input
        # is (batch * num_cams) = batch*6, so a per-GPU batch with batch*6 not
        # divisible by 128 (e.g. 32 -> 192) crashes the first iter. Set
        # im2col_step=6 so ANY batch is divisible (it is a memory-tiling param
        # only, so forward/backward numerics are unchanged).
        for _m in self.model.modules():
            if hasattr(_m, 'im2col_step'):
                _m.im2col_step = 6

        # --- ALL augmentation OFF (fair comparison with BEVFormer) ---
        # IDA: deterministic resize to final_dim, no crop jitter / flip / rotate.
        # resize=0.44 == 704/1600 makes resized width == final_dim width (704),
        # so crop_w=0; this reproduces the eval (is_train=False) transform.
        self.ida_aug_conf = dict(self.ida_aug_conf)
        self.ida_aug_conf.update(resize_lim=(0.44, 0.44), rot_lim=(0.0, 0.0),
                                 rand_flip=False, bot_pct_lim=(0.0, 0.0))
        # BDA: no BEV rotation / scaling / flipping.
        self.bda_aug_conf = dict(self.bda_aug_conf)
        self.bda_aug_conf.update(rot_lim=(0.0, 0.0), scale_lim=(1.0, 1.0),
                                 flip_dx_ratio=0.0, flip_dy_ratio=0.0)
