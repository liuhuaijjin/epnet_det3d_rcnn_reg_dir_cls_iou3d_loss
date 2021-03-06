import torch

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate


class PointIntraPartOffsetHead(PointHeadTemplate):
    """
    Point-based head for predicting the intra-object part locations.
    Reference Paper: https://arxiv.org/abs/1907.03670
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )
        self.part_reg_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.PART_FC,
            input_channels=input_channels,
            output_channels=3
        )
        target_cfg = self.model_cfg.TARGET_CONFIG
        if target_cfg.get('BOX_CODER', None) is not None:
            self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
                **target_cfg.BOX_CODER_CONFIG
            )
            self.box_layers = self.make_fc_layers(
                fc_cfg=self.model_cfg.REG_FC,
                input_channels=input_channels,
                #output_channels=self.box_coder.code_size
                output_channels=8
            )
        else:
            self.box_layers = None

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                backbone_features: (B, C, N)
                backbone_xyz: (B, N, C) [x, y, z]
                gt_boxes3d: (B, M, 7)
        Returns:
            point_cls_labels: (BN), 0:background, -1:ignored, 1:fointground
            point_part_labels: (BN, 3)
        """
        point_coords = input_dict['backbone_xyz'] #[B,N,3]
        gt_boxes = input_dict['gt_boxes3d']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [3], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=True, ret_box_labels=(self.box_layers is not None)
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict = self.get_cls_layer_loss(tb_dict)
        point_loss_part, tb_dict = self.get_part_layer_loss(tb_dict)
        point_loss = point_loss_cls + point_loss_part

        if self.box_layers is not None:
            point_loss_box, tb_dict = self.get_box_layer_loss(tb_dict)
            point_loss += point_loss_box
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                backbone_features: (B, C, N)
                backbone_xyz: (B, N, 3) [x, y, z]
                gt_boxes3d: (B, M, 7)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        point_features = batch_dict['backbone_features'].transpose(1,2).contiguous().view(-1,128) #(BN,128)
        point_cls_preds = self.cls_layers(point_features)  #[BN,1] (total_points, num_class)
        point_part_preds = self.part_reg_layers(point_features) #[BN,3]

        batch_dict['point_cls_preds'] = point_cls_preds
        batch_dict['point_part_preds'] = point_part_preds

        if self.box_layers is not None:
            point_box_preds = self.box_layers(point_features) #[BN,8]
            batch_dict['point_box_preds'] = point_box_preds
        #print(point_cls_preds.shape,point_part_preds.shape,point_box_preds.shape)

        point_cls_scores = torch.sigmoid(point_cls_preds)
        point_part_offset = torch.sigmoid(point_part_preds)
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)
        batch_dict['point_part_offset'] = point_part_offset

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            batch_dict['point_part_labels'] = targets_dict.get('point_part_labels')
            batch_dict['point_box_labels'] = targets_dict.get('point_box_labels')
            #print(targets_dict.get('point_box_labels').shape)

        if self.box_layers is not None and (not self.training or self.predict_boxes_when_training):
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['backbone_xyz'],
                point_cls_preds=point_cls_preds, point_box_preds=batch_dict['point_box_preds']
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

        #self.forward_ret_dict = ret_dict
        return batch_dict
