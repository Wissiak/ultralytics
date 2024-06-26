# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch
import numpy as np

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import CornersMetrics, box_iou


class CornersValidator(DetectionValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize CornersValidator and set task to 'corners', metrics to CornersMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "corners"
        self.metrics = CornersMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)
        # 95% accuracy = if the predicted pixel is withing 10 pixel radius of the ground truth pixel
        radius = 10/640
        self.corner_thresholds = np.array([radius * (10-i) * 14 for i in range(10)])

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6+28] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class and 14 corner points (x,y).
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)
    
    def _process_corners(self, detections, gt_corners, gt_cls):
        pred_classes = detections[:, 5]
        correct_class = gt_cls == pred_classes

        pred_corners = detections[:,6:].reshape(-1, 28)
        pred_corners = pred_corners[correct_class]

        gt_corners = gt_corners.view(-1, 28)

        # sum over distances for each corner
        dist_per_pred = torch.sum(torch.abs(gt_corners.to(pred_corners.device).expand(pred_corners.shape) - pred_corners), 1)

        correct = torch.zeros((detections.shape[0], 0), dtype=torch.bool, device=detections.device)
        for _, thresh in enumerate(self.corner_thresholds):
            corr_i = torch.zeros(detections.shape[0], device=detections.device, dtype=torch.bool)
            corr_i[correct_class] = dist_per_pred < thresh
            correct = torch.cat((correct, corr_i.unsqueeze(1)), 1)

        return correct


    def _prepare_batch(self, si, batch):
        """Prepares and returns a batch for Corners validation."""
        pbatch = super()._prepare_batch(si, batch)
        idx = batch["batch_idx"].cpu() == si
        pbatch["corners"] = batch["corners"][idx]
        #ops.scale_corners(
        #    pbatch["corners"], pbatch["imgsz"]
        #)
        return pbatch

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch for Corners validation with scaled and padded bounding boxes."""
        predn = pred.clone()
        # scale bbox predictions
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
        )  # native-space pred
        # scale corner predictions
        #ops.scale_corners(
        #    predn[:, 6:], pbatch["imgsz"]
        #)

        return predn

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        raise NotImplementedError("TODO: implement plot_predictions")

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        raise NotImplementedError("TODO: implement pred_to_json")

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        raise NotImplementedError("TODO: implement save_one_txt")

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        raise NotImplementedError("TODO: implement eval_json")
