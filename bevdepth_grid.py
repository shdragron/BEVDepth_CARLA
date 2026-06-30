"""BEVDepth qual-grid column for the RoboGeo 4x4 figure. CAM_FRONT clean panels
(GT green + pred red), ego-frame projection (sensor2ego), scene 0267-0016.

Rows:
  pitch12_img : VP IMG (VR=image-only), per-camera CAM_FRONT, pitch +12
  yaw8_img    : VP IMG (VR), all-camera, yaw +8
  suv_cal     : CTS SUV CAL = sedan model on carla_infos_val_suv.pkl (target img+ext)
  bus_cal     : CTS BUS CAL = sedan model on carla_infos_val_bus.pkl
Model input = perturbed/target; display = the image-matching pose (variant for VP,
target for CTS). Run from BEVDepth repo root with the bevdepth env.
"""
import os, os.path as osp, sys, pickle, tempfile
import numpy as np, cv2, torch

from bevdepth.exps.nuscenes.carla.carla_sedan import BEVDepthLightningModel
from bevdepth.datasets.carla_det_dataset import CarlaDetDataset
from bevdepth.datasets.nusc_det_dataset import collate_fn
from _viz_qual import CAM_NAMES, GRID, draw_boxes, in_range, info_key, GT_COLOR, PRED_COLOR
sys.path.insert(0, '/home/hanyan_arch/viewpoint/BEVFormer/bev_det_benchmark')
from build_condition_pkls_bevdepth import make_vp_infos_bevdepth

DATA_ROOT = 'data/carla'
SEDAN_VAL = 'data/carla_infos_val_sedan.pkl'
CKPT = '/home/hanyan_arch/viewpoint/BEVFormer/BEVDepth/outputs/cts_ckpt_sedan.ckpt'
OUT = os.environ.get('OUT', '/home/hanyan_arch/viewpoint/BEVFormer/results/qual_grid')
ROW_SCENE = {'pitch12_img': os.environ.get('SCENE_A', '0230-0032'),
             'yaw8_img':    os.environ.get('SCENE_B', '0256-0120'),
             'suv_cal':     os.environ.get('SCENE_C', '0267-0016'),
             'bus_cal':     os.environ.get('SCENE_C', '0267-0016')}
THR = float(os.environ.get('THR', '0.3'))
CF = CAM_NAMES.index('CAM_FRONT')
os.makedirs(OUT, exist_ok=True)
_TMP = []


def dump_tmp(obj, tag):
    fd, p = tempfile.mkstemp(prefix=f'bdgrid_{tag}_', suffix='.pkl', dir='/tmp'); os.close(fd)
    pickle.dump(obj, open(p, 'wb')); _TMP.append(p); return p


def build_model():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    hp = {k: v for k, v in ck['hyper_parameters'].items() if k not in ('class_names', 'head_conf')}
    m = BEVDepthLightningModel(**hp); m.load_state_dict(ck['state_dict'], strict=False)
    for mod in m.modules():
        if hasattr(mod, 'im2col_step'):
            mod.im2col_step = 6
    return m.cuda().eval()


def build_ds(model, info_path):
    return CarlaDetDataset(
        ida_aug_conf=model.ida_aug_conf, bda_aug_conf=model.bda_aug_conf,
        classes=model.class_names, data_root=DATA_ROOT, info_paths=info_path,
        is_train=True, img_conf=model.img_conf, num_sweeps=model.num_sweeps,
        sweep_idxes=model.sweep_idxes, key_idxes=model.key_idxes,
        return_depth=False, use_fusion=False, gt_visibility_min=model.gt_visibility_min)


def sample_idx(ds, scene, frame):
    frame2 = f'{int(frame) * 2:04d}'
    for i, inf in enumerate(ds.infos):
        sc, fr = info_key(inf)
        if sc == scene and fr in (frame, frame2):
            return i
    raise KeyError(f'{scene}-{frame} not found in {len(ds.infos)} infos')


def run_pred(model, ds, scene, frame):
    item = ds[sample_idx(ds, scene, frame)]
    sweep_imgs, mats, _, img_metas, _, _ = collate_fn([item], is_return_depth=False)
    with torch.no_grad():
        preds = model.model(sweep_imgs.cuda(), {k: v.cuda() for k, v in mats.items()})
        res = model.model.get_bboxes(preds, img_metas)
    return res[0][0].cpu().numpy(), res[0][1].cpu().numpy()


def get_display(ds, scene, frame):
    idx = sample_idx(ds, scene, frame)
    _, mats, _, _, gt_boxes, _ = collate_fn([ds[idx]], is_return_depth=False)
    gt = in_range(gt_boxes[0].cpu().numpy())
    s2e = mats['sensor2ego_mats'][0, 0].cpu().numpy()
    K = mats['intrin_mats'][0, 0].cpu().numpy()
    return gt, s2e, K, ds.infos[idx]


def six_cam(model, mds, dds, scene, frame):
    """2x3 six-camera grid; GT green + pred red via each cam's sensor2ego."""
    pb_all, ps = run_pred(model, mds, scene, frame)
    gt, s2e, K, info = get_display(dds, scene, frame)
    pb = in_range(pb_all[ps >= THR])
    panels = {}
    for ci, cam in enumerate(CAM_NAMES):
        img = cv2.imread(osp.join(DATA_ROOT, info['cam_infos'][cam]['filename']))
        if img is None:
            img = np.zeros((900, 1600, 3), np.uint8)
        draw_boxes(img, gt, s2e[ci], K[ci], GT_COLOR)
        draw_boxes(img, pb, s2e[ci], K[ci], PRED_COLOR)
        panels[cam] = img
    grid = np.vstack([np.hstack([panels[c] for c in row]) for row in GRID])
    return grid, len(gt), len(pb)


def main():
    model = build_model()
    sedan_base = pickle.load(open(SEDAN_VAL, 'rb'))
    disp_normal = build_ds(model, SEDAN_VAL)
    rows = {}
    # VP rows: model-input = VR (image-only), display = CR (variant img+ext)
    for row, axis, mag, proto in [('pitch12_img', 'pitch', 12, 'CAM_FRONT'),
                                  ('yaw8_img', 'yaw', 8, 'all')]:
        vr = make_vp_infos_bevdepth(sedan_base, 'VR', axis, mag, proto)
        cr = make_vp_infos_bevdepth(sedan_base, 'CR', axis, mag, proto)
        rows[row] = (build_ds(model, dump_tmp(vr, f'vr_{row}')),
                     build_ds(model, dump_tmp(cr, f'cr_{row}')))
    # CTS CAL rows: model-input = display = target val pkl (target img+ext)
    for row, tgt in [('suv_cal', 'suv'), ('bus_cal', 'bus')]:
        tds = build_ds(model, f'data/carla_infos_val_{tgt}.pkl')
        rows[row] = (tds, tds)

    for row, (mds, dds) in rows.items():
        scene, frame = ROW_SCENE[row].split('-')
        grid, ng, npd = six_cam(model, mds, dds, scene, frame)
        cv2.imwrite(f'{OUT}/bevdepth_{row}_6cam.png', grid)
        print(f'[bevdepth] {row:12s} scene={scene}-{frame} GT={ng} pred={npd} -> bevdepth_{row}_6cam.png', flush=True)
    for p in _TMP:
        try: os.remove(p)
        except OSError: pass


if __name__ == '__main__':
    main()
