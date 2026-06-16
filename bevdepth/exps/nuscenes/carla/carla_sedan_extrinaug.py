# Copyright (c) Megvii Inc. All rights reserved.
"""BEVDepth on CARLA — sedan, WITH train-time extrinsic-rotation augmentation.

Identical recipe to ``carla_sedan`` (128x128 BEV, EMA off, CBGS off, single-frame,
fp32/TF32-off, 24 ep, lr 2e-4, all IDA/BDA aug off) so it is a controlled A/B vs
the no-aug sedan — the ONLY difference is the extrinsic-noise augmentation:

  with prob 0.5 per sample, each camera's cam->ego extrinsic E is RIGHT-multiplied
  by a random rotation: E' = E @ delta (Euler xyz each ~ U(-20, +20) deg, trans 0),
  i.e. the camera is rotated about its OWN optical center (pan/tilt/roll in place),
  matching how the carla_VR viewpoint variants -- the VP EXT/ER test -- perturb the
  extrinsic (variant = baseline @ delta_cam, position fixed). The image, GT boxes,
  intrinsics and lidar DEPTH GT all stay clean (calibration-noise aug) -> the lift
  sees a mis-calibrated pose, so the model learns to tolerate the VP EXT condition.
  (NB: a left-multiply delta@E would orbit the camera around the ego origin -- a
  DIFFERENT perturbation that the EXT test does not measure, so it gave no EXT gain.)

Run (same as carla_sedan):
  python bevdepth/exps/nuscenes/carla/carla_sedan_extrinaug.py \
      --amp_backend native -b 16 --gpus 2 --accumulate_grad_batches 2
"""
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.carla.carla_base import CarlaBEVDepthBase


class BEVDepthLightningModel(CarlaBEVDepthBase):
    VEHICLE = 'sedan'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # rotation-only ±20 deg, per-camera, Bernoulli(0.5) per sample, trans 0.
        self.extrin_noise_conf = dict(p=0.5, rot_deg=20.0)


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel, 'carla_sedan_extrinaug', use_ema=False,
            extra_trainer_config_args={'precision': 32})
