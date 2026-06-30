"""Condition qualitative BEVDepth CARLA viz, SAME scene across all conditions.

For one car-rich scene, render GT (green) + sedan-model pred (red) 3D boxes on the
6 camera images, one panel per CONDITION:

  VP (viewpoint robustness, sedan model), roll/pitch/yaw @ +20deg, all-6-cams:
     ER  -- extrinsic perturbed, image NOMINAL   (display = normal img + normal ext)
     VR  -- image perturbed, extrinsic nominal    (display = variant img + variant ext)
     CR  -- both perturbed                        (display = variant img + variant ext)
  CTS (cross-platform transfer, sedan model on suv/bus):
     NORMAL -- sedan img + sedan ext              (display = sedan img + sedan ext)
     EXT    -- sedan img + target ext, img NOMINAL(display = sedan img + sedan ext)
     IMG    -- target img + sedan ext             (display = target img + target ext)
     CAL    -- target img + target ext            (display = target img + target ext)

Design (the point the user asked for): the model RUNS on the condition's MODEL INPUT
(perturbed/swapped img+ext) to produce pred boxes in the EGO frame; GT + pred are
then PROJECTED onto the DISPLAY image using the display image's REAL extrinsic (the
camera pose the image was actually rendered with). So GT always lands on the objects,
and pred shows the model's localisation ERROR introduced by the condition -- on a
perturbed image you see pred drift on the (correctly-projected) objects.

Projection + camera order are the verified ones from _viz_qual.py (CAM_NAMES MUST be
the dataset's ida_aug_conf['cams'] order so mats[ci] line up with the image loaded).

Run from the BEVDepth repo root:
    CUDA_VISIBLE_DEVICES=0 python _viz_cond_qual.py
"""
import os
import os.path as osp
import sys
import pickle
import tempfile

import numpy as np
import cv2
import torch

from bevdepth.exps.nuscenes.carla.carla_sedan import BEVDepthLightningModel
from bevdepth.datasets.carla_det_dataset import CarlaDetDataset
from bevdepth.datasets.nusc_det_dataset import collate_fn

# reuse the VERIFIED projection helpers + camera order
from _viz_qual import (CAM_NAMES, GRID, box_corners_ego, draw_boxes, in_range,
                       info_key, GT_COLOR, PRED_COLOR, PC)

sys.path.insert(0, '/home/hanyan_arch/viewpoint/BEVFormer/bev_det_benchmark')
from build_condition_pkls_bevdepth import make_vp_infos_bevdepth     # noqa: E402
# NOTE: vr_image_path in the builder already maps geobev frame N -> carla_VR frame
# 2N (the 1/2-rate relabel fix); sample_idx() below accepts the resulting 2N frame
# token in the rewritten VR/CR filenames.

BEVDEPTH_DATA = 'data'
SEDAN_VAL = 'data/carla_infos_val_sedan.pkl'
CTS_PKL = ('/home/hanyan_arch/viewpoint/BEVFormer/bev_det_benchmark/out/'
           'cts_bevdepth/pkls/{target}_{cond}_infos_val.pkl')
DATA_ROOT = 'data/carla'
CKPT = '/home/hanyan_arch/viewpoint/BEVFormer/BEVDepth/outputs/cts_ckpt_sedan.ckpt'
OUTDIR = '/home/hanyan_arch/viewpoint/BEVFormer/results/BEVDepth/qual_cond'

SCENE, FRAME = '0269', '0150'      # richest in-range scene (23 cars), full VR cover
THRESHOLDS = [0.30, 0.50]
VP_AXES = ['yaw', 'pitch', 'roll']
VP_MAG = 20                         # +20deg only, per the request
CTS_TARGETS = ['suv', 'bus']
CTS_CONDS = ['NORMAL', 'EXT', 'IMG', 'CAL']

_TMP = []                           # keep temp pkls alive for the run


def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def dump_tmp(obj, tag):
    fd, path = tempfile.mkstemp(prefix=f'condqual_{tag}_', suffix='.pkl', dir='/tmp')
    os.close(fd)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    _TMP.append(path)
    return path


def build_model():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    hp = {k: v for k, v in ck['hyper_parameters'].items()
          if k not in ('class_names', 'head_conf')}
    model = BEVDepthLightningModel(**hp)
    model.load_state_dict(ck['state_dict'], strict=False)
    for m in model.modules():
        if hasattr(m, 'im2col_step'):
            m.im2col_step = 6
    return model.cuda().eval()


def build_ds(model, info_path):
    return CarlaDetDataset(
        ida_aug_conf=model.ida_aug_conf, bda_aug_conf=model.bda_aug_conf,
        classes=model.class_names, data_root=DATA_ROOT,
        info_paths=info_path, is_train=True,
        img_conf=model.img_conf, num_sweeps=model.num_sweeps,
        sweep_idxes=model.sweep_idxes, key_idxes=model.key_idxes,
        return_depth=False, use_fusion=False,
        gt_visibility_min=model.gt_visibility_min)


def sample_idx(ds):
    # VR/CR infos rewrite CAM_FRONT's filename to the carla_VR path (frame 2N),
    # so info_key reads frame 2*FRAME for those datasets -- accept either.
    frame2 = f'{int(FRAME) * 2:04d}'
    for i, inf in enumerate(ds.infos):
        sc, fr = info_key(inf)
        if sc == SCENE and fr in (FRAME, frame2):
            return i
    raise KeyError(f'{SCENE}-{FRAME} not in {len(ds.infos)} infos')


def run_pred(model, ds):
    """sedan-model pred boxes (N,9 ego) + scores for the scene's sample."""
    item = ds[sample_idx(ds)]
    batch = collate_fn([item], is_return_depth=False)
    sweep_imgs, mats, _, img_metas, _, _ = batch
    with torch.no_grad():
        preds = model.model(sweep_imgs.cuda(),
                            {k: v.cuda() for k, v in mats.items()})
        res = model.model.get_bboxes(preds, img_metas)
    return res[0][0].cpu().numpy(), res[0][1].cpu().numpy()


def get_display(ds):
    """(gt ego, per-cam s2e, per-cam K, info) for the DISPLAY infos sample."""
    idx = sample_idx(ds)
    item = ds[idx]
    batch = collate_fn([item], is_return_depth=False)
    _, mats, _, _, gt_boxes, _ = batch
    gt = in_range(gt_boxes[0].cpu().numpy())
    s2e = mats['sensor2ego_mats'][0, 0].cpu().numpy()
    K = mats['intrin_mats'][0, 0].cpu().numpy()
    return gt, s2e, K, ds.infos[idx]


def render(disp, pb_all, ps, thr, title, out):
    gt, s2e, K, info = disp
    pb = in_range(pb_all[ps >= thr])
    imgs = {cam: cv2.imread(osp.join(DATA_ROOT, info['cam_infos'][cam]['filename']))
            for cam in CAM_NAMES}
    if any(v is None for v in imgs.values()):
        miss = [c for c, v in imgs.items() if v is None]
        print(f'  SKIP {title} thr{thr}: missing imgs {miss}', flush=True)
        return
    panels = {}
    for ci, cam in enumerate(CAM_NAMES):
        img = imgs[cam].copy()
        draw_boxes(img, gt, s2e[ci], K[ci], GT_COLOR)
        draw_boxes(img, pb, s2e[ci], K[ci], PRED_COLOR)
        cv2.putText(img, cam, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (255, 255, 255), 3, cv2.LINE_AA)
        panels[cam] = img
    h, w = next(iter(panels.values())).shape[:2]
    grid = np.vstack([np.hstack([cv2.resize(panels[c], (w, h)) for c in row])
                      for row in GRID])
    cv2.putText(grid, f'{title}  GT(green)={len(gt)}  pred>{thr:.1f}(red)={len(pb)}',
                (12, grid.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (0, 255, 255), 3, cv2.LINE_AA)
    os.makedirs(osp.dirname(out), exist_ok=True)
    cv2.imwrite(out, grid)
    print(f'  {title} thr{thr:.1f} GT={len(gt)} pred={len(pb)} -> {out}', flush=True)


def main():
    model = build_model()
    sedan_base = load_pkl(SEDAN_VAL)

    # ---- precompute the per-condition (model-input ds, display ds) ----------
    jobs = []   # (group, name, model_ds, display_ds)

    # VP: model runs on perturbed input; display uses the image-matching extrinsic
    disp_normal = build_ds(model, SEDAN_VAL)                      # normal img+ext
    for axis in VP_AXES:
        cr_infos = make_vp_infos_bevdepth(sedan_base, 'CR', axis, VP_MAG, 'all')
        disp_var = build_ds(model, dump_tmp(cr_infos, f'dispvar_{axis}'))  # var img+ext
        er_in = make_vp_infos_bevdepth(sedan_base, 'ER', axis, VP_MAG, 'all')
        vr_in = make_vp_infos_bevdepth(sedan_base, 'VR', axis, VP_MAG, 'all')
        cr_in = cr_infos
        jobs.append(('vp', f'{axis}20_ER', build_ds(model, dump_tmp(er_in, f'er_{axis}')), disp_normal))
        jobs.append(('vp', f'{axis}20_VR', build_ds(model, dump_tmp(vr_in, f'vr_{axis}')), disp_var))
        jobs.append(('vp', f'{axis}20_CR', build_ds(model, dump_tmp(cr_in, f'cr_{axis}')), disp_var))

    # CTS: sedan model on suv/bus condition pkls; display = the image-matching platform
    for target in CTS_TARGETS:
        disp_target = build_ds(model, f'data/carla_infos_val_{target}.pkl')  # target img+ext
        for cond in CTS_CONDS:
            mi = SEDAN_VAL if cond == 'NORMAL' else CTS_PKL.format(target=target, cond=cond)
            display = disp_normal if cond in ('NORMAL', 'EXT') else disp_target
            jobs.append((f'cts_{target}', cond, build_ds(model, mi), display))

    # ---- run model once per (model-input ds), render per threshold -----------
    for group, name, mds, dds in jobs:
        pb_all, ps = run_pred(model, mds)
        disp = get_display(dds)
        for thr in THRESHOLDS:
            title = f'{group}/{name}  scene-{SCENE}-frame-{FRAME}'
            out = osp.join(OUTDIR, f'thr{thr:.1f}', group, f'{name}.jpg')
            render(disp, pb_all, ps, thr, title, out)

    for p in _TMP:
        try:
            os.remove(p)
        except OSError:
            pass


if __name__ == '__main__':
    main()
