# Ultralytics YOLO 🚀, AGPL-3.0 license
from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import CornersModel
from ultralytics.utils import DEFAULT_CFG, RANK


class CornersTrainer(yolo.detect.DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a CornersTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "corners"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return CornersModel initialized with specified config and weights."""
        model = CornersModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of CornersValidator for validation of YOLO model."""
        self.loss_names = "corners_loss", "cls_loss", "dfl_loss" #TODO: adapt loss names
        return yolo.corners.CornersValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
