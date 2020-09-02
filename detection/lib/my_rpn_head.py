import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms

from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from .rpn_test_mixin import RPNTestMixin
from .rpn_head import RPNHead
from mmdet.core import force_fp32
from mmdet.core.bbox.iou_calculators import build_iou_calculator


@HEADS.register_module()
class MyRPNHead(RPNHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels, **kwargs):
        super(MyRPNHead, self).__init__(in_channels, **kwargs)
        self.iou_calculator = build_iou_calculator(dict(type='BboxOverlaps2D'))
        self.bar_loss = build_loss(dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0))
    
    def _init_layers(self):
        super(MyRPNHead, self)._init_layers()
        self.bar_x = nn.Sequential(nn.Conv2d(self.feat_channels, 1, kernel_size=1, bias=False),
                                   nn.Upsample(scale_factor=(1,4)),
                                   # torch.nn.AdaptiveAvgPool2d((1, None)),
                                   torch.nn.AdaptiveMaxPool2d((1, None)),
                                   nn.Sigmoid())
        self.bar_y = nn.Sequential(nn.Conv2d(self.feat_channels, 1, kernel_size=1, bias=False),
                                   nn.Upsample(scale_factor=(4,1)),
                                   # torch.nn.AdaptiveAvgPool2d((None, 1)),
                                   torch.nn.AdaptiveMaxPool2d((None, 1)),
                                   nn.Sigmoid())
    
    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        bar_y = self.bar_x(x)
        bar_x = self.bar_y(x)

        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred, bar_y, bar_x

    def simple_test_rpn(self, x, img_metas, gt_labels=None, gt_bboxes=None):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image.
        """
        rpn_cls_score, rpn_bbox_pred, bar_y, bar_x = self(x)
        proposal_list = self.get_bboxes(rpn_cls_score, rpn_bbox_pred, bar_y, bar_x, img_metas, gt_bboxes)

        return proposal_list
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bar_x_preds', 'bar_y_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             bar_x_preds,
             bar_y_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        losses = super(MyRPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore
        )
        # calculate targets for bars
        H, W = bar_y_preds[0].shape[2], bar_x_preds[0].shape[3]
        bar_x_targets, bar_y_targets = self.get_bar_targets((H, W), gt_bboxes)
        
        # calculate loss weight for each element
        bar_x_w = torch.zeros_like(bar_x_preds[0])
        bar_y_w = torch.zeros_like(bar_y_preds[0])
        #
        bar_x_w[bar_x_targets==1] = 1 - bar_x_targets.mean()
        bar_x_w[bar_x_targets==0] = bar_x_targets.mean()
        #
        bar_y_w[bar_y_targets==1] = 1 - bar_y_targets.mean()
        bar_y_w[bar_y_targets==0] = bar_y_targets.mean()
        
        bar_x_loss = torch.nn.functional.binary_cross_entropy(bar_x_preds[0], bar_x_targets, weight=bar_x_w)
        bar_y_loss = torch.nn.functional.binary_cross_entropy(bar_y_preds[0], bar_y_targets, weight=bar_y_w)
        # bar_x_loss = self.bar_loss(bar_x_preds[0], bar_x_targets)
        # bar_y_loss = self.bar_loss(bar_y_preds[0], bar_y_targets)
        return dict(
            loss_rpn_cls=losses['loss_rpn_cls'], loss_rpn_bbox=losses['loss_rpn_bbox'],
            bar_x_loss=bar_x_loss, bar_y_loss=bar_y_loss)
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   bar_x_preds,
                   bar_y_preds,
                   img_metas,
                   gt_bboxes=None,
                   cfg=None,
                   rescale=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                bar_x_preds[0][img_id].squeeze(), bar_y_preds[0][img_id].squeeze(),
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale, gt_bboxes=gt_bboxes[img_id] if gt_bboxes else None)
            result_list.append(proposals)
        return result_list

    def get_bar_targets(self, size, gt_bboxes):
        device = gt_bboxes[0].device
        H, W = size
        bs = len(gt_bboxes)
        bar_y_targets = torch.zeros(bs, 1, H, 1, device=device)
        bar_x_targets = torch.zeros(bs, 1, 1, W, device=device)

        for i in range(bs):
            gt_bboxes_ = gt_bboxes[i].round().long()
            for j in range(gt_bboxes_.shape[0]):
                x1, y1, x2, y2 = gt_bboxes_[j, ...] - 1
                bar_x_targets[i, 0, 0, x1] = 1
                bar_x_targets[i, 0, 0, x2] = 1
                bar_y_targets[i, 0, y1, 0] = 1
                bar_y_targets[i, 0, y2, 0] = 1
        return bar_x_targets, bar_y_targets


    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """

        labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result \
          = super(MyRPNHead, self)._get_targets_single(flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, \
          gt_labels, img_meta, label_channels, unmap_outputs)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)


    def _get_bboxes_single(self,
                           cls_scores, bbox_preds,
                           bar_x_preds, bar_y_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False, gt_bboxes=None):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # we set FG labels to [0, num_class-1] and BG label to
                # num_class in other heads since mmdet v2.0, However we
                # keep BG label as 0 and FG label as 1 in rpn head
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        # #
        # if gt_bboxes is not None:
        #     gt_bboxes = gt_bboxes.squeeze(dim=0)
        #     if gt_bboxes.shape[0] > 0:
        #         pre_max = max(scores.max(), 1e-5)
        #         iou = self.iou_calculator(proposals, gt_bboxes)
        #         iou, _ = torch.max(iou, dim=1)
        #         scores = scores * iou
        #         r = scores.max() / pre_max
        #         scores /= r
        # #

        pre_max = max(scores.max(), 1e-5)
        proposals_ = proposals.round().long() - 1
        tl_x = bar_x_preds[proposals_[:, 0]]
        tl_y = bar_y_preds[proposals_[:, 1]]
        br_x = bar_x_preds[proposals_[:, 2]]
        br_y = bar_y_preds[proposals_[:, 3]]

        scores *= tl_x * tl_y * br_x * br_y

        r = scores.max() / pre_max
        scores /= r

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        # TODO: remove the hard coded nms type
        nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
        dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
        return dets[:cfg.nms_post]
