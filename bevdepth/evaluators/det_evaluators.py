'''Modified from # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
'''
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

__all__ = ['DetNuscEvaluator']

# CARLA GT-validity rule for eval: keep annotations with visibility_token >= this.
# Matches training (CarlaDetDataset.gt_visibility_min=2) and BEVFormer's
# visibility>=2 valid_flag, instead of the devkit's default num_pts>0 rule.
CARLA_MIN_VISIBILITY = 2


class DetNuscEvaluator():
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE',
    }

    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }

    def __init__(
        self,
        class_names,
        eval_version='detection_cvpr_2019',
        data_root='./data/nuScenes',
        version='v1.0-trainval',
        modality=dict(use_lidar=False,
                      use_camera=True,
                      use_radar=False,
                      use_map=False,
                      use_external=False),
        output_dir=None,
    ) -> None:
        self.eval_version = eval_version
        self.data_root = data_root
        if self.eval_version is not None:
            from nuscenes.eval.detection.config import config_factory

            self.eval_detection_configs = config_factory(self.eval_version)
        self.version = version
        self.class_names = class_names
        self.modality = modality
        self.output_dir = output_dir

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(version=self.version,
                        dataroot=self.data_root,
                        verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        if self.version in eval_set_map:
            nusc_eval = NuScenesEval(nusc,
                                     config=self.eval_detection_configs,
                                     result_path=result_path,
                                     eval_set=eval_set_map[self.version],
                                     output_dir=output_dir,
                                     verbose=False)
            nusc_eval.main(render_curves=False)
        else:
            # --- CARLA custom version (e.g. v1.0-carla_sedan_eval) ----------
            # The stock devkit only knows v1.0-{mini,trainval,test} and filters
            # GT by num_lidar_pts+num_radar_pts>0 (the nuScenes "valid" rule).
            # CARLA instead (a) uses a custom version string + scene names and
            # (b) trains on a *visibility*-based valid set (gt_visibility_min=2),
            # NOT num_pts -- and BEVFormer's pkl valid_flag is likewise
            # visibility>=2. To make eval match TRAINING (and stay comparable to
            # BEVFormer) we swap in a custom load_gt that:
            #   1. restricts GT to exactly the predicted scenes (so the devkit
            #      pred==gt sample_tokens assertion holds -- the eval DB may hold
            #      scenes val.txt drops, e.g. scene_0260);
            #   2. keeps only GT with visibility_token >= CARLA_MIN_VISIBILITY
            #      (== training's gt_visibility_min), replacing the num_pts rule;
            #   3. forces num_pts=1 so the devkit's num_pts==0 removal in
            #      filter_eval_boxes is a no-op (visibility is now the sole
            #      validity rule). class_range distance filtering is unchanged.
            # Only evaluate.load_gt is patched (scoped to this construction);
            # the rest is stock NuScenesEval, so the 6-class NDS is computed
            # identically to BEVFormer.
            assert 'carla' in self.version, \
                f'Unknown dataset version for eval: {self.version!r}'
            import tqdm as _tqdm
            from nuscenes.eval.common.data_classes import EvalBoxes
            from nuscenes.eval.detection.utils import category_to_detection_name
            from nuscenes.eval.detection import evaluate as _eval_mod

            pred_tokens = set(mmcv.load(result_path)['results'].keys())
            scene_tokens = {nusc.get('sample', t)['scene_token']
                            for t in pred_tokens}
            carla_scenes = {nusc.get('scene', st)['name']
                            for st in scene_tokens}
            print(f'[CARLA-EVAL] version={self.version} eval_split=val '
                  f'min_visibility>={CARLA_MIN_VISIBILITY} '
                  f'pred_samples={len(pred_tokens)} '
                  f'carla_scenes={len(carla_scenes)}', flush=True)
            attribute_map = {a['token']: a['name'] for a in nusc.attribute}

            def _carla_load_gt(nusc_, eval_split, box_cls, verbose=False):
                """GT loader keeping visibility_token >= CARLA_MIN_VISIBILITY
                (training parity), num_pts forced to 1, restricted to EXACTLY the
                predicted samples. Using pred_tokens (not whole scenes) makes the
                devkit pred==gt assertion hold for both the full eval set and a
                frames-per-scene subset (VP)."""
                gt = EvalBoxes()
                for st in _tqdm.tqdm(sorted(pred_tokens), leave=verbose):
                    boxes = []
                    for at in nusc_.get('sample', st)['anns']:
                        a = nusc_.get('sample_annotation', at)
                        dn = category_to_detection_name(a['category_name'])
                        if dn is None:
                            continue
                        if int(a['visibility_token']) < CARLA_MIN_VISIBILITY:
                            continue
                        attr = a['attribute_tokens']
                        attribute_name = (attribute_map[attr[0]]
                                          if len(attr) == 1 else '')
                        boxes.append(box_cls(
                            sample_token=st,
                            translation=a['translation'], size=a['size'],
                            rotation=a['rotation'],
                            velocity=nusc_.box_velocity(a['token'])[:2],
                            num_pts=1,
                            detection_name=dn, detection_score=-1.0,
                            attribute_name=attribute_name))
                    gt.add_boxes(st, boxes)
                return gt

            _orig_load_gt = _eval_mod.load_gt
            _eval_mod.load_gt = _carla_load_gt
            try:
                nusc_eval = NuScenesEval(nusc,
                                         config=self.eval_detection_configs,
                                         result_path=result_path,
                                         eval_set='val',
                                         output_dir=output_dir,
                                         verbose=False)
            finally:
                _eval_mod.load_gt = _orig_load_gt
            nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for class_name in self.class_names:
            for k, v in metrics['label_aps'][class_name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, class_name,
                                                 k)] = val
            for k, v in metrics['label_tp_errors'][class_name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, class_name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        # Recompute mAP/NDS over exactly self.class_names using the devkit
        # formula. For nuScenes (10 classes) this reproduces the devkit numbers;
        # for CARLA (6 classes) it gives a clean score not diluted by the 4
        # absent detection_cvpr_2019 classes (which score AP=0). Matches
        # BEVFormer's CARLA eval. NDS = (5*mAP + sum max(0,1-nanmean(TP)))/10.
        import numpy as np
        tp_keys = list(metrics['tp_errors'].keys())
        mean_dist_aps = [np.mean(list(metrics['label_aps'][c].values()))
                         for c in self.class_names]
        mAPc = float(np.mean(mean_dist_aps))
        tp_scores = [max(0.0, 1.0 - float(np.nanmean(
            [metrics['label_tp_errors'][c][m] for c in self.class_names])))
            for m in tp_keys]
        ndsc = (5.0 * mAPc + float(np.sum(tp_scores))) / (5.0 + len(tp_keys))
        detail['{}/NDS'.format(metric_prefix)] = ndsc
        detail['{}/mAP'.format(metric_prefix)] = mAPc
        detail['{}/NDS_allclass'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP_allclass'.format(metric_prefix)] = metrics['mean_ap']
        # 6-class TP-error components (nanmean over self.class_names) -- the
        # ingredients of ndsc -- and *_10class aliases, so the bev_det_benchmark
        # CTS driver populates its mATE..mAAE / NDS_10class / mAP_10class
        # columns (key-name parity with BEVFormer's CarlaNuScenesDataset).
        for m in tp_keys:
            err6 = float(np.nanmean(
                [metrics['label_tp_errors'][c][m] for c in self.class_names]))
            detail['{}/{}_6class'.format(
                metric_prefix, self.ErrNameMapping[m])] = round(err6, 4)
        detail['{}/NDS_10class'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP_10class'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self,
                       results,
                       img_metas,
                       result_names=['img_bbox'],
                       jsonfile_prefix=None,
                       **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
        result_files = dict()
        # refactor this.
        for rasult_name in result_names:
            # not evaluate 2D predictions on nuScenes
            if '2d' in rasult_name:
                continue
            print(f'\nFormating bboxes of {rasult_name}')
            tmp_file_ = osp.join(jsonfile_prefix, rasult_name)
            if self.output_dir:
                result_files.update({
                    rasult_name:
                    self._format_bbox(results, img_metas, self.output_dir)
                })
            else:
                result_files.update({
                    rasult_name:
                    self._format_bbox(results, img_metas, tmp_file_)
                })
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        img_metas,
        metric='bbox',
        logger=None,
        jsonfile_prefix=None,
        result_names=['img_bbox'],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, img_metas,
                                                    result_names,
                                                    jsonfile_prefix)
        detail = None
        if isinstance(result_files, dict):
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                detail = self._evaluate_single(result_files[name])
        elif isinstance(result_files, str):
            detail = self._evaluate_single(result_files)

        # Surface the recomputed N-class (CARLA: 6) mAP/NDS on stdout so the
        # bev_det_benchmark drivers can scrape it (same format as BEVFormer's
        # tools/test.py [CARLA-EVAL] line). detail otherwise only lives in the
        # discarded return value above.
        if detail is not None:
            import json as _json
            nds = next((v for k, v in detail.items()
                        if k.endswith('/NDS')), None)
            mapc = next((v for k, v in detail.items()
                         if k.endswith('/mAP')), None)
            if nds is not None and mapc is not None:
                print('[CARLA-EVAL] {}-class mAP={:.4f} NDS={:.4f}'.format(
                    len(self.class_names), mapc, nds), flush=True)
                print('[CARLA-METRICS-JSON] ' + _json.dumps(detail), flush=True)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return detail

    def _format_bbox(self, results, img_metas, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.class_names

        print('Start to convert detection format...')

        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes, scores, labels = det
            boxes = boxes
            sample_token = img_metas[sample_id]['token']
            trans = np.array(img_metas[sample_id]['ego2global_translation'])
            rot = Quaternion(img_metas[sample_id]['ego2global_rotation'])
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = Box(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                if np.sqrt(nusc_box.velocity[0]**2 +
                           nusc_box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    velocity=nusc_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
