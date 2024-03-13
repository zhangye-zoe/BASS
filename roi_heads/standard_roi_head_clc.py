import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import os
from copy import deepcopy
from ilib.utils import seg2edge

@HEADS.register_module()
class StandardRoIHeadCLC(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                bbox_roi_extractor=None,
                bbox_head=None,
                mask_roi_extractor=None,
                mask_head=None,
                shared_head=None,
                train_cfg=None,
                test_cfg=None,
                pretrained=None,
                init_cfg=None,
                contrastive_roi_extractor=None):
        super(StandardRoIHeadCLC, self).__init__(
                bbox_roi_extractor=bbox_roi_extractor,
                bbox_head=bbox_head,
                mask_roi_extractor=mask_roi_extractor,
                mask_head=mask_head,
                shared_head=shared_head,
                train_cfg=train_cfg,
                test_cfg=test_cfg,
                pretrained=pretrained,
                init_cfg=init_cfg)

        if contrastive_roi_extractor is not None:
            self.contrastive_roi_extractor = build_roi_extractor(contrastive_roi_extractor)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        import pickle
        #self.nn = pickle.load(open('nn.pkl','rb'))
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_bboxes_scores=None,
                      gt_masks_scores=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas,)
            losses.update(mask_results['loss_mask'])
        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                      gt_labels, self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        bbox_results.update(bbox_targets = bbox_targets)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
            img_metas,):
        """Run forward function and calculate loss for mask head in
        training."""

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks, self.train_cfg)
        edge_targets = seg2edge(mask_targets)
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            # print("pos rois", pos_rois)
            mask_results = self._mask_forward(x, pos_rois, edges=edge_targets, masks=mask_targets)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        

        train_cfg2 = deepcopy(self.train_cfg)
        train_cfg2['mask_size'] = 14 ###
        mask_targets_coarse = self.mask_head.get_targets(sampling_results, gt_masks, train_cfg2)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        import numpy as np
        np.save("z_mask_target.npy", mask_targets.cpu().numpy())
        np.save("z_mask_coarse_target.npy", mask_targets_coarse.cpu().numpy())
        print("mask target", mask_targets.shape)
        print("mask coarse target", mask_targets_coarse.shape)
        print("pos labels", pos_labels.shape)



        loss_mask = self.mask_head.loss(mask_results['mask_pred'], mask_results['mask_pred_coarse'], #mask_results['mask_pred_coarse2'],######
                                        mask_targets, mask_targets_coarse,
                                        pos_labels)

        contrastive_sets = mask_results["contrastive_sets"]
        mask_pred = mask_results['mask_pred']
        if contrastive_sets is not None:
            loss_contrastive = self.mask_head.contrastive_head.loss(contrastive_sets['sample_easy_pos'],
                                                                    contrastive_sets['sample_easy_neg'],
                                                                    contrastive_sets['sample_hard_pos'],
                                                                    contrastive_sets['sample_hard_neg'],
                                                                    contrastive_sets['query_pos'],
                                                                    contrastive_sets['query_neg'],
                                                                    t_easy=0.3,
                                                                    t_hard=0.7)
            loss_mask['loss_mask'] = loss_mask['loss_mask'] + 1 * loss_contrastive
            # print("loss mask", loss_mask)
            # print("loss constractive",loss_contrastive)
            # loss_mask['weight_for_cl'] = torch.tensor(self.mask_head.contrastive_head.weight).cuda()
            # loss_mask['pred_num_all'] = torch.tensor(float(mask_pred.shape[0])).cuda()
            # loss_mask['pred_num_base'] = torch.tensor(float(mask_pred[valid_masks].shape[0])).cuda()
        
        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None, edges=None, masks=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)

            if self.mask_head.contrastive_enable:
                contrastive_feats = self.contrastive_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], rois
                )
            else:
                contrastive_feats = None

            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)

        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred, mask_pred_coarse, contrastive_sets = self.mask_head(mask_feats, 
                                                        contrastive_feats,
                                                        edges,
                                                        masks)
        mask_results = dict(mask_pred=mask_pred, mask_pred_coarse=mask_pred_coarse, mask_feats=mask_feats, contrastive_sets=contrastive_sets)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]
