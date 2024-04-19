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
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "c1_x_loss", "c1_y_loss", "c2_x_loss", "c2_y_loss", "c3_x_loss", "c3_y_loss", "c4_x_loss", "c4_y_loss", "c5_x_loss", "c5_y_loss", "c6_x_loss", "c6_y_loss", "c7_x_loss", "c7_y_loss", "c8_x_loss", "c8_y_loss", "c9_x_loss", "c9_y_loss", "c10_x_loss", "c10_y_loss", "c11_x_loss", "c11_y_loss", "c12_x_loss", "c12_y_loss", "c13_x_loss", "c13_y_loss", "c14_x_loss", "c14_y_loss"
        return yolo.corners.CornersValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
