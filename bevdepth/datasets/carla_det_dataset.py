# Copyright (c) Megvii Inc. All rights reserved.
"""CARLA detection dataset.

A ``NuscDetDataset`` variant for the nuScenes-format CARLA data. It keeps all
the nuScenes loading/augmentation logic and only overrides the pieces that
differ for CARLA:

* lidar points are stored as ``.npz`` (key ``'data'``, xyz only) instead of the
  nuScenes 5-field ``.bin``;
* GT boxes are filtered by ``visibility_token >= gt_visibility_min`` instead of
  ``num_lidar_pts + num_radar_pts > 0`` (when ``gt_visibility_min`` is set);
* depth supervision comes from the per-vehicle rendered DPT depth image (dense)
  instead of the sparse lidar projection (when ``depth_gt_from_image`` is True).
"""
import os

import numpy as np
import torch
from PIL import Image

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn

__all__ = ['CarlaDetDataset', 'collate_fn']


def decode_carla_depth(dpt_png_path):
    """Decode a CARLA depth PNG (RGB-encoded) to a dense HxW depth map in meters.

    CARLA depth encoding: normalized = (R + G*256 + B*256^2) / (256^3 - 1),
    depth_m = normalized * 1000. (sky / far clip -> ~1000 m).
    """
    arr = np.asarray(Image.open(dpt_png_path), dtype=np.float64)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    norm = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0**3 - 1.0)
    return (norm * 1000.0).astype(np.float32)


def depth_image_transform(depth, resize_dims, crop, flip, rotate):
    """Apply the SAME IDA geometry as img_transform to a dense depth map.

    NEAREST resampling so depth edges are not blended. Returns a torch float
    tensor of shape final_dim (H, W) = (crop[3]-crop[1], crop[2]-crop[0]).
    """
    d = Image.fromarray(depth, mode='F')
    d = d.resize(resize_dims, resample=Image.NEAREST)
    d = d.crop(crop)
    if flip:
        d = d.transpose(method=Image.FLIP_LEFT_RIGHT)
    d = d.rotate(rotate, resample=Image.NEAREST)
    return torch.from_numpy(np.array(d, dtype=np.float32))


class CarlaDetDataset(NuscDetDataset):

    # Use the dense rendered DPT depth image for depth supervision (per
    # vehicle), instead of the sparse lidar projection.
    depth_gt_from_image = True

    def _load_lidar_points(self, full_path):
        if full_path.endswith('.npz'):
            # CARLA lidar: npz with key 'data', shape (N, 3) xyz only.
            # LidarPointCloud expects 4 dims, so pad intensity=0.
            pts = np.load(full_path)['data'].astype(np.float32)
            return np.concatenate(
                [pts, np.zeros((pts.shape[0], 1), dtype=np.float32)], 1)
        return super()._load_lidar_points(full_path)

    def _keep_ann(self, ann_info):
        if self.gt_visibility_min is not None:
            # visibility-based filter: keep boxes with visibility_token >= min.
            return int(ann_info['visibility_token']) >= self.gt_visibility_min
        return super()._keep_ann(ann_info)

    def _get_cam_depth(self, cam_info, img, lidar_info, lidar_points, resize,
                       resize_dims, crop, flip, rotate):
        if not self.depth_gt_from_image:
            return super()._get_cam_depth(cam_info, img, lidar_info,
                                          lidar_points, resize, resize_dims,
                                          crop, flip, rotate)
        # Dense depth GT from the rendered DPT image for this camera/vehicle.
        # The RGB path encodes the vehicle (RGB-CAM_FRONT=sedan, RGB-bus-*=bus,
        # RGB-suv-*=suv); the DPT path is derived 1:1 (RGB->DPT, .jpg->.png).
        dpt_path = os.path.join(
            self.data_root,
            cam_info['filename'].replace('RGB', 'DPT').replace('.jpg', '.png'))
        depth = decode_carla_depth(dpt_path)
        # Out-of-range depths (incl. ~1000 m sky) are masked out later by
        # get_downsampled_gt_depth's d_bound binning, like empty lidar pixels.
        return depth_image_transform(depth, resize_dims, crop, flip, rotate)
