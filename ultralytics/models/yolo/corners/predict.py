# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class CornersPredictor(DetectionPredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes CornersPredictor with optional model and data configuration overrides."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "corners"

    def postprocess(self, preds, img, orig_imgs):
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )
