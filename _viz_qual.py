"""Qualitative BEVDepth CARLA viz: project GT (green) + predicted (red) 3D boxes
onto the original 6 camera images, per vehicle (sedan/suv/bus), for the highest
IN-RANGE-object val samples. Output: a 2x3 six-view grid per (vehicle, sample).

Projection (verified vs nuScenes Box.render_cv2): boxes are in the global-axes
ego frame (the frame sensor2ego maps to); pixel = K @ inv(sensor2ego) @ corners,
drawn on the ORIGINAL image (intrinsic principal point == image centre, so no IDA).
Only in-range (|x|,|y|<PC) boxes whose centre is in front and inside the camera
FOV are drawn -- this culls the wild near-camera-plane / off-edge projections.

Run from the BEVDepth repo root:
    CUDA_VISIBLE_DEVICES=0 python _viz_qual.py
"""
import os
import os.path as osp
import re
import itertools
from functools import partial

import numpy as np
import cv2
import torch

from bevdepth.exps.nuscenes.carla.carla_sedan import BEVDepthLightningModel
from bevdepth.datasets.carla_det_dataset import CarlaDetDataset
from bevdepth.datasets.nusc_det_dataset import collate_fn

# MUST match the dataset's camera order (ida_aug_conf['cams']) so mats[ci] line up
# with the image we load -- BEVDepth uses this order, NOT the FRONT-first one.
CAM_NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
GRID = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]
SCENE_FRAME_RE = re.compile(r'scene-(\d+)-frame-(\d+)')

# Top-10 distinct-scene samples by IN-pc-range (vis>=2) object count.
SAMPLES = [('0245', '0108'), ('0247', '0142'), ('0241', '0066'), ('0244', '0050'),
           ('0222', '0126'), ('0246', '0142'), ('0267', '0026'), ('0266', '0040'),
           ('0230', '0000'), ('0269', '0150')]
VEHICLES = ['sedan', 'suv', 'bus']
CKPT = '/home/hanyan_arch/viewpoint/BEVFormer/BEVDepth/outputs/cts_ckpt_{}.ckpt'
DATA_ROOT = 'data/carla'
OUTDIR = '/home/hanyan_arch/viewpoint/BEVFormer/results/BEVDepth/qual'
THRESHOLDS = [0.30, 0.50]    # produce a qual set per pred-score threshold
PC = 51.2                     # detection / point-cloud range (xy half-extent)
EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
         (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
GT_COLOR = (0, 230, 0)       # green (BGR)
PRED_COLOR = (0, 0, 255)     # red   (BGR)


def info_key(s):
    return SCENE_FRAME_RE.search(s['cam_infos']['CAM_FRONT']['filename']).groups()


def in_range(boxes):
    if len(boxes) == 0:
        return boxes
    return boxes[(np.abs(boxes[:, 0]) < PC) & (np.abs(boxes[:, 1]) < PC)]


def box_corners_ego(box):
    x, y, z, dx, dy, dz, yaw = box[:7]
    c = np.array([(sx * dx / 2, sy * dy / 2, sz * dz / 2)
                  for sx, sy, sz in itertools.product([1, -1], [1, -1], [1, -1])])
    cs, sn = np.cos(yaw), np.sin(yaw)
    R = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
    return (R @ c.T).T + np.array([x, y, z])


def draw_boxes(img, boxes, s2e, K, color):
    """Draw only boxes whose centre is in front of and inside this camera's FOV,
    with every corner ahead of the near plane -- cull the wild edge projections."""
    e2c = np.linalg.inv(s2e)
    R, t = e2c[:3, :3], e2c[:3, 3]
    K3 = K[:3, :3]
    H, W = img.shape[:2]
    for box in boxes:
        ctr = R @ box[:3] + t                          # centre in cam frame
        if ctr[2] < 1.5 or ctr[2] > 70:                # too close/behind or too far
            continue
        cu = K3 @ ctr
        cx, cy = cu[0] / cu[2], cu[1] / cu[2]
        if not (-100 < cx < W + 100 and -100 < cy < H + 100):   # centre off-image
            continue
        cam = (R @ box_corners_ego(box).T).T + t       # (8,3) corners in cam
        if (cam[:, 2] <= 0.3).any():                   # a corner behind near plane
            continue
        uvw = (K3 @ cam.T).T
        uv = (uvw[:, :2] / uvw[:, 2:3]).astype(int)
        for i, j in EDGES:
            cv2.line(img, tuple(uv[i]), tuple(uv[j]), color, 2, cv2.LINE_AA)


def build_model():
    ck = torch.load(CKPT.format('sedan'), map_location='cpu', weights_only=False)
    hp = {k: v for k, v in ck['hyper_parameters'].items()
          if k not in ('class_names', 'head_conf')}
    model = BEVDepthLightningModel(**hp)
    for m in model.modules():
        if hasattr(m, 'im2col_step'):
            m.im2col_step = 6
    return model.cuda().eval()


def build_ds(model, veh):
    return CarlaDetDataset(
        ida_aug_conf=model.ida_aug_conf, bda_aug_conf=model.bda_aug_conf,
        classes=model.class_names, data_root=DATA_ROOT,
        info_paths=f'data/carla_infos_val_{veh}.pkl', is_train=True,
        img_conf=model.img_conf, num_sweeps=model.num_sweeps,
        sweep_idxes=model.sweep_idxes, key_idxes=model.key_idxes,
        return_depth=False, use_fusion=False, gt_visibility_min=model.gt_visibility_min)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    model = build_model()
    for veh in VEHICLES:
        ck = torch.load(CKPT.format(veh), map_location='cpu', weights_only=False)
        model.load_state_dict(ck['state_dict'], strict=False)
        model.eval()
        ds = build_ds(model, veh)
        key2idx = {info_key(inf): i for i, inf in enumerate(ds.infos)}
        for scene, frame in SAMPLES:
            if (scene, frame) not in key2idx:
                print(f'[{veh}] MISSING {scene}-{frame}', flush=True)
                continue
            idx = key2idx[(scene, frame)]
            item = ds[idx]
            batch = collate_fn([item], is_return_depth=False)
            sweep_imgs, mats, _, img_metas, gt_boxes, gt_labels = batch
            with torch.no_grad():
                preds = model.model(sweep_imgs.cuda(),
                                    {k: v.cuda() for k, v in mats.items()})
                res = model.model.get_bboxes(preds, img_metas)
            pb_all = res[0][0].cpu().numpy()
            ps = res[0][1].cpu().numpy()
            gb = in_range(gt_boxes[0].cpu().numpy())
            s2e = mats['sensor2ego_mats'][0, 0].cpu().numpy()
            K = mats['intrin_mats'][0, 0].cpu().numpy()
            info = ds.infos[idx]
            # load each camera's image ONCE (CAM_NAMES order == mats order)
            base_imgs = {cam: cv2.imread(osp.join(DATA_ROOT, info['cam_infos'][cam]['filename']))
                         for cam in CAM_NAMES}
            if any(v is None for v in base_imgs.values()):
                continue
            for thr in THRESHOLDS:
                pb = in_range(pb_all[ps >= thr])
                panels = {}
                for ci, cam in enumerate(CAM_NAMES):
                    img = base_imgs[cam].copy()
                    draw_boxes(img, gb, s2e[ci], K[ci], GT_COLOR)
                    draw_boxes(img, pb, s2e[ci], K[ci], PRED_COLOR)
                    cv2.putText(img, cam, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                                (255, 255, 255), 3, cv2.LINE_AA)
                    panels[cam] = img
                h, w = next(iter(panels.values())).shape[:2]
                grid = np.vstack([np.hstack([cv2.resize(panels[c], (w, h)) for c in row])
                                  for row in GRID])
                cv2.putText(grid, f'{veh}  scene-{scene}-frame-{frame}  '
                            f'GT(green,in-range)={len(gb)}  pred>{thr:.1f}(red)={len(pb)}',
                            (12, grid.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                            (0, 255, 255), 3, cv2.LINE_AA)
                out = osp.join(OUTDIR, f'thr{thr:.1f}',
                               f'{veh}_scene-{scene}-frame-{frame}.jpg')
                os.makedirs(osp.dirname(out), exist_ok=True)
                cv2.imwrite(out, grid)
                print(f'[{veh}] {scene}-{frame} thr{thr:.1f} GT={len(gb)} '
                      f'pred={len(pb)} -> {out}', flush=True)


if __name__ == '__main__':
    main()
