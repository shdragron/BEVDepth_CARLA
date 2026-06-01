"""Generate BEVDepth *train-split* info pkl for the CARLA dataset.

The CARLA data at CARLA_ROOT is stored as nuScenes DB tables but uses its own
scene names (scene_XXXX) and has all timestamps == 0, so the stock
``scripts/gen_info.py`` (hard-coded to nuScenes splits + ``box_velocity``)
cannot be used directly.

Split parity with BEVFormer: the train/val membership comes from the shared
``<CARLA_ROOT>/split/{train,val}.txt`` (one scene name per line), NOT a
VAL_EVERY heuristic. train.txt == the v1.0-carla_<veh> train DB (220 scenes), so
this script writes every train-DB scene that is listed in train.txt.

Usage:
  python scripts/gen_info_carla.py --version v1.0-carla_sedan --tag sedan
  python scripts/gen_info_carla.py --version v1.0-carla_bus   --tag bus
"""
import argparse
import os

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

CARLA_ROOT = '/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev'
CARLA_VERSION = 'v1.0-carla_sedan'
OUT_DIR = './data'


def load_split_scenes(name):
    """Read scene names from <CARLA_ROOT>/split/<name>.txt ('train' or 'val')."""
    path = os.path.join(CARLA_ROOT, 'split', f'{name}.txt')
    with open(path) as f:
        return {ln.strip() for ln in f if ln.strip()}


def generate_info(nusc, scenes, max_cam_sweeps=6, max_lidar_sweeps=10):
    infos = list()
    for cur_scene in tqdm(nusc.scene):
        if cur_scene['name'] not in scenes:
            continue
        first_sample_token = cur_scene['first_sample_token']
        cur_sample = nusc.get('sample', first_sample_token)
        while True:
            info = dict()
            cam_datas = list()
            lidar_datas = list()
            info['sample_token'] = cur_sample['token']
            info['timestamp'] = cur_sample['timestamp']
            info['scene_token'] = cur_sample['scene_token']
            cam_names = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
            ]
            lidar_names = ['LIDAR_TOP']
            cam_infos = dict()
            lidar_infos = dict()
            for cam_name in cam_names:
                cam_data = nusc.get('sample_data',
                                    cur_sample['data'][cam_name])
                cam_datas.append(cam_data)
                sweep_cam_info = dict()
                sweep_cam_info['sample_token'] = cam_data['sample_token']
                sweep_cam_info['ego_pose'] = nusc.get('ego_pose',
                                                      cam_data['ego_pose_token'])
                sweep_cam_info['timestamp'] = cam_data['timestamp']
                sweep_cam_info['is_key_frame'] = cam_data['is_key_frame']
                sweep_cam_info['height'] = cam_data['height']
                sweep_cam_info['width'] = cam_data['width']
                sweep_cam_info['filename'] = cam_data['filename']
                sweep_cam_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', cam_data['calibrated_sensor_token'])
                cam_infos[cam_name] = sweep_cam_info
            for lidar_name in lidar_names:
                lidar_data = nusc.get('sample_data',
                                      cur_sample['data'][lidar_name])
                lidar_datas.append(lidar_data)
                sweep_lidar_info = dict()
                sweep_lidar_info['sample_token'] = lidar_data['sample_token']
                sweep_lidar_info['ego_pose'] = nusc.get(
                    'ego_pose', lidar_data['ego_pose_token'])
                sweep_lidar_info['timestamp'] = lidar_data['timestamp']
                sweep_lidar_info['filename'] = lidar_data['filename']
                sweep_lidar_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', lidar_data['calibrated_sensor_token'])
                lidar_infos[lidar_name] = sweep_lidar_info

            info['cam_infos'] = cam_infos
            info['lidar_infos'] = lidar_infos
            # CARLA frames have no intra-sample sweeps (prev/next empty on
            # sample_data), so sweep lists stay empty -> single-frame only.
            info['cam_sweeps'] = list()
            info['lidar_sweeps'] = list()
            ann_infos = list()
            if 'anns' in cur_sample:
                for ann in cur_sample['anns']:
                    ann_info = nusc.get('sample_annotation', ann)
                    # CARLA has real 5 Hz timestamps -> box_velocity is valid
                    # (matches BEVFormer / nuScenes). Keep it (nan -> 0).
                    ann_info['velocity'] = np.nan_to_num(
                        nusc.box_velocity(ann_info['token']), nan=0.0)
                    ann_infos.append(ann_info)
                info['ann_infos'] = ann_infos
            infos.append(info)
            if cur_sample['next'] == '':
                break
            else:
                cur_sample = nusc.get('sample', cur_sample['next'])
    return infos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default=CARLA_VERSION,
                        help='CARLA train DB version, e.g. v1.0-carla_sedan')
    parser.add_argument('--tag', default='',
                        help="output suffix; 'sedan' -> "
                        'carla_infos_train_sedan.pkl')
    args = parser.parse_args()
    suffix = f'_{args.tag}' if args.tag else ''

    nusc = NuScenes(version=args.version, dataroot=CARLA_ROOT, verbose=True)
    train_scenes = load_split_scenes('train')
    db_scenes = {s['name'] for s in nusc.scene}
    use_scenes = train_scenes & db_scenes
    dropped = db_scenes - train_scenes
    print(f'{args.version}: db_scenes={len(db_scenes)} '
          f'train.txt={len(train_scenes)} using={len(use_scenes)} '
          f'dropped_from_db={sorted(dropped) if dropped else 0}')

    train_infos = generate_info(nusc, use_scenes)
    print(f'train samples: {len(train_infos)}')

    mmcv.mkdir_or_exist(OUT_DIR)
    out = os.path.join(OUT_DIR, f'carla_infos_train{suffix}.pkl')
    mmcv.dump(train_infos, out)
    print('wrote', out)


if __name__ == '__main__':
    main()
