# Copyright (c) Megvii Inc. All rights reserved.
"""BEVDepth on CARLA — sedan (subcompact) viewpoint.

Run:
  python bevdepth/exps/nuscenes/carla/carla_sedan.py --amp_backend native -b 8 --gpus 2
"""
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.carla.carla_base import CarlaBEVDepthBase


class BEVDepthLightningModel(CarlaBEVDepthBase):
    VEHICLE = 'sedan'


if __name__ == '__main__':
    # precision=32 (fp32) + EMA off to match BEVFormer (fair comparison).
    run_cli(BEVDepthLightningModel, 'carla_sedan', use_ema=False,
            extra_trainer_config_args={'precision': 32})
