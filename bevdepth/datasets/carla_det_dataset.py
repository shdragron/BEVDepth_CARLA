# Copyright (c) Megvii Inc. All rights reserved.
"""CARLA detection dataset.

A ``NuscDetDataset`` variant for the nuScenes-format CARLA data. It keeps all
the nuScenes loading/augmentation logic and only overrides the two pieces that
differ for CARLA:

* lidar points are stored as ``.npz`` (key ``'data'``, xyz only) instead of the
  nuScenes 5-field ``.bin``;
* GT boxes are filtered by ``visibility_token >= gt_visibility_min`` instead of
  ``num_lidar_pts + num_radar_pts > 0`` (when ``gt_visibility_min`` is set).
"""
import numpy as np

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn

__all__ = ['CarlaDetDataset', 'collate_fn']


class CarlaDetDataset(NuscDetDataset):

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
