# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self.use_clothes = cfg.MODEL.LOSSES.USE_CLOTHES

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        # backbone
        self.backbone = build_backbone(cfg)
        # head
        self.heads = build_heads(cfg)
        # Train with clothes ids
        if self.use_clothes:
            self.clo_heads = build_heads(cfg, True)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)
            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()
            outputs = self.heads(features, targets)

            if self.use_clothes:
                assert "clothids" in batched_inputs, "Clothes ID annotation are missing in training!"
                targets_clo = batched_inputs['clothids'].to(self.device)
                if targets_clo.sum() < 0: targets_clo.zero_()
                outputs_clo = self.clo_heads(features, targets_clo)
            else:
                targets_clo = None
                outputs_clo = None

            return {
                "outputs": outputs,
                "targets": targets,
                "outputs_clo": outputs_clo,
                "targets_clo": targets_clo,
            }

        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))
        
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # import pdb; pdb.set_trace()
        # fmt: off
        outputs           = outs["outputs"]
        gt_labels         = outs["targets"]
        # model predictions for persons
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']

        if self.use_clothes:
            # model prediction for clothes
            outputs_clo       = outs["outputs_clo"]
            gt_clo_labels     = outs["targets_clo"]

            pred_clo_class_logits = outputs_clo['pred_class_logits'].detach()
            cls_clo_outputs = outputs_clo['cls_outputs']

        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CE.SCALE

        # import pdb; pdb.set_trace()
        # calculate cloth id loss
        if self.use_clothes:
            loss_dict['loss_clo_cls'] = cross_entropy_loss(
                    cls_clo_outputs,
                    gt_clo_labels,
                    self._cfg.MODEL.LOSSES.CE.EPSILON,
                    self._cfg.MODEL.LOSSES.CE.ALPHA,
                ) * self._cfg.MODEL.LOSSES.CE.SCALE

        if "TripletLoss" in loss_names:
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE

        if "CircleLoss" in loss_names:
            loss_dict['loss_circle'] = circle_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                self._cfg.MODEL.LOSSES.CIRCLE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CIRCLE.SCALE
        

        return loss_dict
