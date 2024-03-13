import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
from prettytable import PrettyTable
import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
import scipy.io as sio

from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.coco import CocoDataset
from ..utils import (pre_eval_all_semantic_metric, pre_eval_bin_aji, pre_eval_bin_pq, pre_eval_to_sem_metrics,
                         pre_eval_to_imw_sem_metrics, pre_eval_aji, pre_eval_pq, pre_eval_to_bin_aji, pre_eval_to_aji,
                         pre_eval_to_bin_pq, pre_eval_to_pq, pre_eval_to_imw_pq, pre_eval_to_imw_aji)

from ..utils import re_instance, assign_sem_class_to_insts

# DATASET for partially supervised mode in coco
@DATASETS.register_module()
class CocoPSDataset(CocoDataset):
    # NOVEL_CLASSES = {
    #     'voc':(
    #         0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62
    #     ),
    #     'nonvoc':(
    #         7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    #         38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65, 66,
    #         67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79
    #     ),
    # }
    NOVEL_CLASSES= {
        'voc': (0,),
        'nonvoc': (),

    }

    def __init__(self,
                 ann_file,
                 pipeline,
                 ann_dir,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                #  ann_dir='/data4/zhangye/noisyboundaries/data/CryoNuSeg/test/mask',
                 inst_suffix='.mat',
                 base_set = 'nonvoc',
                 novel_set='voc'):
        super(CocoPSDataset, self).__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            ann_dir=ann_dir,
            classes=None,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt)
        self.novel_set = novel_set
        self.base_set = base_set
        self.novel_set_labels = self.NOVEL_CLASSES[novel_set]
        self.base_set_labels = self.NOVEL_CLASSES[base_set]


    def load_annotations(self, ann_file, ann_dir, inst_suffix):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.objid2label = {obj['id']: i for i, obj in enumerate(self.coco.dataset['annotations'])} 
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            # print("info", info)
            img_id = info['file_name'].split(".")[0]
            # print("img id", img_id)
            # print("suffix", inst_suffix)
            inst_name = img_id + inst_suffix
            info['filename'] = info['file_name']
            # info['sem_file_name'] = osp.join(ann_dir, sem_name)
            # print("ann dir", ann_dir)
            # print("inst name", inst_name)
            info['inst_file_name'] = osp.join(ann_dir, inst_name)
            # print("info", info['inst_file_name'])
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info, ann_ids) 

    def _parse_ann_info(self, img_info, ann_info, ann_ids):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_bboxes_ids = []
        gt_labels = []
        gt_labels_names = []
        is_novel = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_bboxes_ids.append(ann_ids[i]) 
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_labels_names.append(self.CLASSES[self.cat2label[ann['category_id']]])
                is_novel.append(self.cat2label[ann['category_id']] not in \
                                self.base_set_labels)
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            is_novel = np.array(is_novel, dtype=np.int64)
            gt_bboxes_ids = np.array(gt_bboxes_ids, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            is_novel = np.array([], dtype=np.int64)
            gt_bboxes_ids = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            bboxes_ids=gt_bboxes_ids,
            labels=gt_labels,
            labels_names=gt_labels_names,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            is_novel=is_novel)
        
        # print("ann", ann["masks"])
        # print("=" * 100)
        
        return ann

    def prepare_test_img(self, idx):
        """Get testing data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)



    

    # def evaluate(self,
    #              results,
    #              metric='bbox',
    #              logger=None,
    #              jsonfile_prefix=None,
    #              classwise=False,
    #              proposal_nums=(100, 300, 1000),
    #              iou_thrs=None,
    #              metric_items=None):

    #     # print("result", results)
    #     # print("+" * 100)

    #     metrics = metric if isinstance(metric, list) else [metric]
    #     allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
    #     for metric in metrics:
    #         if metric not in allowed_metrics:
    #             raise KeyError(f'metric {metric} is not supported')
    #     if iou_thrs is None:
    #         iou_thrs = np.linspace(
    #             .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    #     if metric_items is not None:
    #         if not isinstance(metric_items, list):
    #             metric_items = [metric_items]
    #     # print("results", results)
    #     result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        
    #     # print("result files", result_files)
    #     # print("tmp dir", tmp_dir)
    #     # print("="* 100)
    #     eval_results = OrderedDict()
    #     cocoGt = self.coco

    #     # print("dataset", len(self.coco.dataset["annotations"]))
    #     # print("=" * 100)

    #     for metric in metrics:
    #         msg = f'Evaluating {metric}...'
    #         if logger is None:
    #             msg = '\n' + msg
    #         print_log(msg, logger=logger)

    #         if metric == 'proposal_fast':
    #             ar = self.fast_eval_recall(
    #                 results, proposal_nums, iou_thrs, logger='silent')
    #             log_msg = []
    #             for i, num in enumerate(proposal_nums):
    #                 eval_results[f'AR@{num}'] = ar[i]
    #                 log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
    #             log_msg = ''.join(log_msg)
    #             print_log(log_msg, logger=logger)
    #             continue

    #         iou_type = 'bbox' if metric == 'proposal' else metric
    #         if metric not in result_files:
    #             raise KeyError(f'{metric} is not in results')
    #         try:
    #             predictions = mmcv.load(result_files[metric])
    #             if iou_type == 'segm':
    #                 # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
    #                 # When evaluating mask AP, if the results contain bbox,
    #                 # cocoapi will use the box area instead of the mask area
    #                 # for calculating the instance area. Though the overall AP
    #                 # is not affected, this leads to different
    #                 # small/medium/large mask AP results.
    #                 for x in predictions:
    #                     x.pop('bbox')
    #                 warnings.simplefilter('once')
    #                 warnings.warn(
    #                     'The key "bbox" is deleted for more accurate mask AP '
    #                     'of small/medium/large instances since v2.12.0. This '
    #                     'does not change the overall mAP calculation.',
    #                     UserWarning)
    #             # print("predictions", predictions[0])

    #             cocoDt = cocoGt.loadRes(predictions)
    #             # print("cocoDt", cocoDt.dataset)
    #             # print("=" * 100)
    #         except IndexError:
    #             print_log(
    #                 'The testing results of the whole dataset is empty.',
    #                 logger=logger,
    #                 level=logging.ERROR)
    #             break

    #         cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            

    #         # cocoEval.params.catIds = self.cat_ids
    #         # only evaluate novel cat_ids
    #         novel_cat_ids = [self.cat_ids[label] for label in self.novel_set_labels]
    #         cocoEval.params.catIds = novel_cat_ids
    #         cocoEval.params.imgIds = self.img_ids
    #         cocoEval.params.maxDets = list(proposal_nums)
    #         cocoEval.params.iouThrs = iou_thrs
    #         # mapping of cocoEval.stats
    #         coco_metric_names = {
    #             'mAP': 0,
    #             'mAP_50': 1,
    #             'mAP_75': 2,
    #             'mAP_s': 3,
    #             'mAP_m': 4,
    #             'mAP_l': 5,
    #             'AR@100': 6,
    #             'AR@300': 7,
    #             'AR@1000': 8,
    #             'AR_s@1000': 9,
    #             'AR_m@1000': 10,
    #             'AR_l@1000': 11
    #         }
    #         if metric_items is not None:
    #             for metric_item in metric_items:
    #                 if metric_item not in coco_metric_names:
    #                     raise KeyError(
    #                         f'metric item {metric_item} is not supported')

    #         if metric == 'proposal':
    #             cocoEval.params.useCats = 0
    #             cocoEval.evaluate()
    #             cocoEval.accumulate()
    #             cocoEval.summarize()
    #             if metric_items is None:
    #                 metric_items = [
    #                     'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
    #                     'AR_m@1000', 'AR_l@1000'
    #                 ]

    #             for item in metric_items:
    #                 val = float(
    #                     f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
    #                 eval_results[item] = val
    #         else:
    #             cocoEval.evaluate()
    #             cocoEval.accumulate()
    #             cocoEval.summarize()
    #             if classwise:  # Compute per-category AP
    #                 # Compute per-category AP
    #                 # from https://github.com/facebookresearch/detectron2/
    #                 precisions = cocoEval.eval['precision']
    #                 # print("precision", precisions)
    #                 # print("=" * 100)
    #                 # precision: (iou, recall, cls, area range, max dets)
    #                 # assert len(self.cat_ids) == precisions.shape[2]
    #                 assert len(novel_cat_ids) == precisions.shape[2]

    #                 results_per_category = []
    #                 for idx, catId in enumerate(novel_cat_ids):
    #                     # area range index 0: all area ranges
    #                     # max dets index -1: typically 100 per image
    #                     nm = self.coco.loadCats(catId)[0]
    #                     precision = precisions[:, :, idx, 0, -1]
    #                     precision = precision[precision > -1]
    #                     if precision.size:
    #                         ap = np.mean(precision)
    #                     else:
    #                         ap = float('nan')
    #                     results_per_category.append(
    #                         (f'{nm["name"]}', f'{float(ap):0.3f}'))

    #                 num_columns = min(6, len(results_per_category) * 2)
    #                 results_flatten = list(
    #                     itertools.chain(*results_per_category))
    #                 headers = ['category', 'AP'] * (num_columns // 2)
    #                 results_2d = itertools.zip_longest(*[
    #                     results_flatten[i::num_columns]
    #                     for i in range(num_columns)
    #                 ])
    #                 table_data = [headers]
    #                 table_data += [result for result in results_2d]
    #                 table = AsciiTable(table_data)
    #                 print_log('\n' + table.table, logger=logger)

    #             if metric_items is None:
    #                 metric_items = [
    #                     'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
    #                 ]

    #             for metric_item in metric_items:
    #                 key = f'{metric}_{metric_item}'
    #                 val = float(
    #                     f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
    #                 )
    #                 eval_results[key] = val
    #             ap = cocoEval.stats[:6]
    #             eval_results[f'{metric}_mAP_copypaste'] = (
    #                 f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
    #                 f'{ap[4]:.3f} {ap[5]:.3f}')
    #     if tmp_dir is not None:
    #         tmp_dir.cleanup()
    #     return eval_results
        
    # """

    def pre_eval(self, preds, indices, show=False, show_folder='.nuclei_show'):

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            # sem_file_name = self.data_infos[index]['sem_file_name']
            # # semantic level label make
            # sem_gt = mmcv.imread(sem_file_name, flag='unchanged', backend='pillow')
            # instance level label make
            inst_file_name = self.data_infos[index]['inst_file_name'].split('.')[0]
            # inst_gt = np.load(inst_file_name)
            inst_gt = sio.loadmat(inst_file_name)["inst_map"]
            inst_gt = re_instance(inst_gt)

            sem_gt = (inst_gt>0).astype(np.uint8)

            # metric calculation & post process codes:
            sem_pred = (pred['sem_pred'].copy()/255).astype(np.uint8)
            inst_pred = pred['inst_pred'].copy()
            # print("sem pred", np.unique(sem_pred))
            # print("sem gt", np.unique(sem_gt))
            # print("=" * 100)

            # semantic metric calculation (remove background class)
            sem_pre_eval_res = pre_eval_all_semantic_metric(sem_pred, sem_gt, 2)

            # make contiguous ids
            inst_pred = re_instance(inst_pred)
            inst_gt = re_instance(inst_gt)

            # print("inst pred", inst_pred)
            # print("inst gt", inst_gt)
            # print("=" * 100)
            # print("num class", self.CLASSES)

            pred_id_list_per_class = assign_sem_class_to_insts(inst_pred, sem_pred, 2)
            gt_id_list_per_class = assign_sem_class_to_insts(inst_gt, sem_gt, 2)

            # instance metric calculation
            aji_pre_eval_res = pre_eval_aji(inst_pred, inst_gt, pred_id_list_per_class, gt_id_list_per_class,
                                            2)
            # print("aji pre eval res", aji_pre_eval_res)
            # print("=" * 100)
            
            bin_aji_pre_eval_res = pre_eval_bin_aji(inst_pred, inst_gt)

            pq_pre_eval_res = pre_eval_pq(inst_pred, inst_gt, pred_id_list_per_class, gt_id_list_per_class,
                                          2)
            bin_pq_pre_eval_res = pre_eval_bin_pq(inst_pred, inst_gt)

            single_loop_results = dict(
                bin_aji_pre_eval_res=bin_aji_pre_eval_res,
                aji_pre_eval_res=aji_pre_eval_res,
                bin_pq_pre_eval_res=bin_pq_pre_eval_res,
                pq_pre_eval_res=pq_pre_eval_res,
                sem_pre_eval_res=sem_pre_eval_res)
            pre_eval_results.append(single_loop_results)

            # print("metrics", single_loop_results)

        return pre_eval_results





    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the dataset.
        Args:
            processor (object): The result processor.
            metric (str | list[str]): Metrics to be evaluated. 'Aji',
                'Dice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            dump_path (str | None, optional): The dump path of each item
                evaluation results. Default: None
        Returns:
            dict[str, float]: Default metrics.
        """

        # print("results", results[0])
        # print("=" * 100)

        ret_metrics = {}
        img_ret_metrics = {}

        # list to dict
        for result in results:
            print("result", result)
            print("=" * 100)
            for key, value in result[0].items():
                if key not in ret_metrics:
                    ret_metrics[key] = [value]
                else:
                    ret_metrics[key].append(value)

        # semantic metrics
        sem_pre_eval_results = ret_metrics.pop('sem_pre_eval_res')
        ret_metrics.update(pre_eval_to_sem_metrics(sem_pre_eval_results, metrics=['Dice', 'Precision', 'Recall']))
        img_ret_metrics.update(
            pre_eval_to_imw_sem_metrics(sem_pre_eval_results, metrics=['Dice', 'Precision', 'Recall']))

        # aji metrics
        aji_pre_eval_results = ret_metrics.pop('aji_pre_eval_res')
        bin_aji_pre_eval_results = ret_metrics.pop('bin_aji_pre_eval_res')
        ret_metrics.update(pre_eval_to_aji(aji_pre_eval_results))
        for k, v in pre_eval_to_bin_aji(bin_aji_pre_eval_results).items():
            ret_metrics['b' + k] = v
        img_ret_metrics.update(pre_eval_to_imw_aji(bin_aji_pre_eval_results))

        # pq metrics
        pq_pre_eval_results = ret_metrics.pop('pq_pre_eval_res')
        bin_pq_pre_eval_results = ret_metrics.pop('bin_pq_pre_eval_res')
        ret_metrics.update(pre_eval_to_pq(pq_pre_eval_results))
        for k, v in pre_eval_to_bin_pq(bin_pq_pre_eval_results).items():
            ret_metrics['b' + k] = v
        img_ret_metrics.update(pre_eval_to_imw_pq(bin_pq_pre_eval_results))

        vital_keys = ['Dice', 'Precision', 'Recall', 'Aji', 'DQ', 'SQ', 'PQ']
        mean_metrics = {}
        overall_metrics = {}
        classes_metrics = OrderedDict()
        # calculate average metric
        for key in vital_keys:
            # XXX: Using average value may have lower metric value than using
            # confused matrix.
            mean_metrics['imw' + key] = np.nanmean(img_ret_metrics[key])
            overall_metrics['m' + key] = np.nanmean(ret_metrics[key])
            # class wise metric
            classes_metrics[key] = ret_metrics[key]
            average_value = np.nanmean(classes_metrics[key])
            tmp_list = classes_metrics[key].tolist()
            tmp_list.append(average_value)
            classes_metrics[key] = np.array(tmp_list)

        for key in ['bAji', 'bDQ', 'bSQ', 'bPQ']:
            overall_metrics[key] = ret_metrics[key]

        # class wise table
        classes_metrics.update(
            OrderedDict({class_key: np.round(value * 100, 2)
                         for class_key, value in classes_metrics.items()}))
        # classes_metrics.update(OrderedDict({analysis_key: value for analysis_key, value in inst_analysis.items()}))

        # remove background class
        classes_metrics.update({'classes': list(self.CLASSES[0:]) + ['average']})
        classes_metrics.move_to_end('classes', last=False)

        classes_table_data = PrettyTable()
        for key, val in classes_metrics.items():
            classes_table_data.add_column(key, val)

        print_log('Per classes:', logger)
        print_log('\n' + classes_table_data.get_string(), logger=logger)

        # mean table
        mean_metrics = OrderedDict({key: np.round(np.mean(value) * 100, 2) for key, value in mean_metrics.items()})

        mean_table_data = PrettyTable()
        for key, val in mean_metrics.items():
            mean_table_data.add_column(key, [val])

        # overall table
        overall_metrics = OrderedDict(
            {key: np.round(np.mean(value) * 100, 2)
             for key, value in overall_metrics.items()})

        overall_table_data = PrettyTable()
        for key, val in overall_metrics.items():
            overall_table_data.add_column(key, [val])

        print_log('Mean Total:', logger)
        print_log('\n' + mean_table_data.get_string(), logger=logger)
        print_log('Overall Total:', logger)
        print_log('\n' + overall_table_data.get_string(), logger=logger)

        storage_results = {
            'mean_metrics': mean_metrics,
            'overall_metrics': overall_metrics,
        }

        eval_results = {}
        for k, v in overall_metrics.items():
            eval_results[k] = v
        for k, v in mean_metrics.items():
            eval_results[k] = v

        classes = classes_metrics.pop('classes', None)
        for key, value in classes_metrics.items():
            eval_results.update({key + '.' + str(name): f'{value[idx]:.3f}' for idx, name in enumerate(classes)})

        # This ret value is used for eval hook. Eval hook will add these
        # evaluation info to runner.log_buffer.output. Then when the
        # TextLoggerHook is called, the evaluation info will output to logger.
        return eval_results, storage_results


def re_instance(instance_map):
    """convert sparse instance ids to continual instance ids for instance
    map."""
    instance_ids = list(np.unique(instance_map))
    instance_ids.remove(0) if 0 in instance_ids else None
    new_instance_map = np.zeros_like(instance_map, dtype=np.int32)

    for id, instance_id in enumerate(instance_ids):
        new_instance_map[instance_map == instance_id] = id + 1

    return new_instance_map