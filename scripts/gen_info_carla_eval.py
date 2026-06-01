"""Generate BEVDepth *eval-split* info pkl for the CARLA dataset.

The CARLA dataset ships dedicated evaluation DBs (``v1.0-carla_<vehicle>_eval``)
whose scenes do not overlap the training DB. This script loads such an eval DB
and writes the val split.

Split parity with BEVFormer: only scenes listed in ``<CARLA_ROOT>/split/val.txt``
are kept (48 scenes). The eval DB has 49 scenes; the one not in val.txt
(scene_0260) is dropped so the eval set matches BEVFormer exactly.

Usage:
  python scripts/gen_info_carla_eval.py --version v1.0-carla_sedan_eval --tag sedan
  python scripts/gen_info_carla_eval.py --version v1.0-carla_bus_eval   --tag bus
"""
import argparse
import os

import mmcv
from nuscenes.nuscenes import NuScenes

from gen_info_carla import (CARLA_ROOT, OUT_DIR, generate_info,
                            load_split_scenes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v1.0-carla_sedan_eval',
                        help='CARLA eval DB version')
    parser.add_argument('--tag', default='',
                        help="output suffix; 'sedan' -> carla_infos_val_sedan.pkl")
    args = parser.parse_args()
    suffix = f'_{args.tag}' if args.tag else ''

    nusc = NuScenes(version=args.version, dataroot=CARLA_ROOT, verbose=True)
    val_scenes = load_split_scenes('val')
    db_scenes = {s['name'] for s in nusc.scene}
    use_scenes = val_scenes & db_scenes
    dropped = db_scenes - val_scenes
    print(f'{args.version}: db_scenes={len(db_scenes)} '
          f'val.txt={len(val_scenes)} using={len(use_scenes)} '
          f'dropped_from_db={sorted(dropped) if dropped else 0}')

    val_infos = generate_info(nusc, use_scenes)
    print(f'val samples: {len(val_infos)}')

    mmcv.mkdir_or_exist(OUT_DIR)
    out = os.path.join(OUT_DIR, f'carla_infos_val{suffix}.pkl')
    mmcv.dump(val_infos, out)
    print('wrote', out)


if __name__ == '__main__':
    main()
