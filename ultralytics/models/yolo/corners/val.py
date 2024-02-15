# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import CornersMetrics, batch_probiou
from ultralytics.utils.plotting import output_to_rotated_target, plot_images


class CornersValidator(DetectionValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize CornersValidator and set task to 'corners', metrics to CornersMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "corners"
        self.metrics = CornersMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.is_dota = isinstance(val, str) and "DOTA" in val  # is COCO

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
            rotated=True,
        )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def _prepare_batch(self, si, batch):
        """Prepares and returns a batch for Corners validation."""
        raise NotImplementedError("TODO: implement _prepare_batch")

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch for Corners validation with scaled and padded bounding boxes."""
        raise NotImplementedError("TODO: implement _prepare_pred")

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
