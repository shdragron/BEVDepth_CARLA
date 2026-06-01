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

from bevdepth.datasets.carla_det_dataset import CarlaDetDataset
from bevdepth.exps.nuscenes.base_exp import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
from bevdepth.exps.nuscenes.base_exp import head_conf

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
        # Evaluator reads GT from the per-vehicle eval DB.
        self.evaluator.data_root = CARLA_DB_ROOT
        self.evaluator.version = f'v1.0-carla_{veh}_eval'
        # visibility >= 2 GT filter (matches BEVFormer); CarlaDetDataset applies
        # it via _keep_ann, replacing the default num_lidar_pts+radar>0 filter.
        self.gt_visibility_min = 2
        # Use the CARLA dataset (npz lidar loader + visibility filter).
        self.dataset_class = CarlaDetDataset

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
