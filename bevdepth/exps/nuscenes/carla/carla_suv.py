# Copyright (c) Megvii Inc. All rights reserved.
"""BEVDepth on CARLA — suv viewpoint.

Run:
  python bevdepth/exps/nuscenes/carla/carla_suv.py --amp_backend native -b 8 --gpus 2
"""
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.carla.carla_base import CarlaBEVDepthBase


class BEVDepthLightningModel(CarlaBEVDepthBase):
    VEHICLE = 'suv'


if __name__ == '__main__':
    # precision=32 (fp32) + EMA off to match BEVFormer (fair comparison).
    run_cli(BEVDepthLightningModel, 'carla_suv', use_ema=False,
            extra_trainer_config_args={'precision': 32})
