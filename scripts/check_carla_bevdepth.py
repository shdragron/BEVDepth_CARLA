"""Coordinate-system check for BEVDepth on CARLA.

Verifies, using BEVDepth's OWN transforms, that:
  1. GT boxes from CarlaDetDataset.get_gt (ego frame, lwh + raw yaw) project onto
     the camera images and land on objects.
  2. The lidar depth supervision (map_pointcloud_to_image: lidar->ego->global->
     cam) projects points onto the image consistent with scene geometry.

Renders <out>/<sample>_<cam>.jpg with GT boxes (green) + depth points (jet).

Run from BEVDepth/:
  PYTHONPATH=. python scripts/check_carla_bevdepth.py --pkl data/carla_infos_val_sedan.pkl
"""
import argparse
import os
import pickle

import cv2
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from bevdepth.datasets.carla_det_dataset import CarlaDetDataset
from bevdepth.datasets.nusc_det_dataset import map_pointcloud_to_image

CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
             'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
CARLA_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian']


def ego_box_corners(box):
    """box = [x,y,z, dx(l), dy(w), dz(h), yaw, ...] in ego frame -> 8x3."""
    x, y, z, dx, dy, dz, yaw = box[:7]
    xc = np.array([dx, dx, -dx, -dx, dx, dx, -dx, -dx]) / 2
    yc = np.array([dy, -dy, -dy, dy, dy, -dy, -dy, dy]) / 2
    zc = np.array([-dz, -dz, -dz, -dz, dz, dz, dz, dz]) / 2
    corners = np.stack([xc, yc, zc], 0)  # 3x8
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = R @ corners
    corners += np.array([[x], [y], [z]])
    return corners.T  # 8x3


def ego_to_cam_pixels(pts_ego, cam_cs):
    """pts_ego: Nx3 in ego frame. Returns (uv Nx2, depth N) via cam calib."""
    pts = pts_ego.T.copy()  # 3xN
    pts = pts - np.array(cam_cs['translation'])[:, None]
    pts = Quaternion(cam_cs['rotation']).rotation_matrix.T @ pts
    depth = pts[2, :]
    uv = view_points(pts, np.array(cam_cs['camera_intrinsic']), normalize=True)
    return uv[:2].T, depth


def draw_box(img, pts, color=(0, 255, 0)):
    p = pts.astype(int)
    for j in range(4):
        cv2.line(img, tuple(p[j]), tuple(p[(j + 1) % 4]), color, 2)
        cv2.line(img, tuple(p[j + 4]), tuple(p[4 + (j + 1) % 4]), color, 1)
        cv2.line(img, tuple(p[j]), tuple(p[j + 4]), color, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='data/carla_infos_val_sedan.pkl')
    ap.add_argument('--data-root', default='data/carla')
    ap.add_argument('--out', default='data/_bevdepth_check')
    ap.add_argument('--idxs', default='0,1200,2400,3600')
    ap.add_argument('--cams', default='CAM_FRONT,CAM_BACK')
    args = ap.parse_args()

    infos = pickle.load(open(args.pkl, 'rb'))
    print(f'infos: {len(infos)}')
    os.makedirs(args.out, exist_ok=True)

    gt = CarlaDetDataset.__new__(CarlaDetDataset)
    gt.classes = CARLA_CLASSES
    gt.gt_visibility_min = 2

    idxs = [int(x) for x in args.idxs.split(',') if x.strip()]
    cams = args.cams.split(',')
    for i in idxs:
        info = infos[i]
        boxes, labels = gt.get_gt(info, CAM_NAMES)  # ego frame, vis>=2 filtered
        boxes = boxes.numpy() if hasattr(boxes, 'numpy') else np.array(boxes)
        tok = info['sample_token'][:8]
        for cam in cams:
            ci = info['cam_infos'][cam]
            img_path = os.path.join(args.data_root, ci['filename'])
            if not os.path.isfile(img_path):
                print('missing', img_path)
                continue
            img = cv2.imread(img_path)
            H, W = img.shape[:2]

            # --- (b) lidar depth projection (BEVDepth's own function) ---
            li = info['lidar_infos']['LIDAR_TOP']
            lpts = np.load(os.path.join(args.data_root, li['filename']))['data']
            lpts = np.concatenate(
                [lpts.astype(np.float32),
                 np.zeros((lpts.shape[0], 1), np.float32)], 1)
            from PIL import Image
            pil = Image.open(img_path)
            ppts, depth = map_pointcloud_to_image(
                lpts, pil, li['calibrated_sensor'], li['ego_pose'],
                ci['calibrated_sensor'], ci['ego_pose'])
            uvd = ppts[:2].T.astype(int)
            for (u, v), d in zip(uvd, depth):
                t = min(d, 60.0) / 60.0
                col = (int(255 * t), int(120), int(255 * (1 - t)))  # far blue->near red
                cv2.circle(img, (u, v), 1, col, -1)
            n_depth = len(depth)

            # --- (a) GT box projection (ego -> cam) ---
            n_drawn = 0
            for b in boxes:
                corners = ego_box_corners(b)
                uv, dz = ego_to_cam_pixels(corners, ci['calibrated_sensor'])
                if (dz > 0.5).all() and \
                   (uv[:, 0] > -2000).all() and (uv[:, 0] < W + 2000).all():
                    cen, cd = ego_to_cam_pixels(b[None, :3], ci['calibrated_sensor'])
                    if cd[0] > 0.5 and 0 <= cen[0, 0] < W and 0 <= cen[0, 1] < H:
                        draw_box(img, uv)
                        n_drawn += 1
            out = os.path.join(args.out, f'{tok}_{cam}.jpg')
            cv2.imwrite(out, img)
            print(f'idx={i} {cam}: gt_boxes={len(boxes)} drawn={n_drawn} '
                  f'depth_pts={n_depth} -> {out}')


if __name__ == '__main__':
    main()
