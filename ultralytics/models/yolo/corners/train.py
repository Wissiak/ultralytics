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
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "c1_x_loss", "c1_y_loss", "c2_x_loss", "c2_y_loss", "c3_x_loss", "c3_y_loss", "c4_x_loss", "c4_y_loss", "c5_x_loss", "c5_y_loss", "c6_x_loss", "c6_y_loss", "c7_x_loss", "c7_y_loss", "c8_x_loss", "c8_y_loss", "c9_x_loss", "c9_y_loss", "c10_x_loss", "c10_y_loss", "c11_x_loss", "c11_y_loss", "c12_x_loss", "c12_y_loss", "c13_x_loss", "c13_y_loss", "c14_x_loss", "c14_y_loss", "c15_x_loss", "c15_y_loss", "c16_x_loss", "c16_y_loss", "c17_x_loss", "c17_y_loss", "c18_x_loss", "c18_y_loss", "c19_x_loss", "c19_y_loss", "c20_x_loss", "c20_y_loss", "c21_x_loss", "c21_y_loss", "c22_x_loss", "c22_y_loss", "c23_x_loss", "c23_y_loss", "c24_x_loss", "c24_y_loss", "c25_x_loss", "c25_y_loss", "c26_x_loss", "c26_y_loss", "c27_x_loss", "c27_y_loss", "c28_x_loss", "c28_y_loss", "c29_x_loss", "c29_y_loss", "c30_x_loss", "c30_y_loss", "c31_x_loss", "c31_y_loss", "c32_x_loss", "c32_y_loss", "c33_x_loss", "c33_y_loss", "c34_x_loss", "c34_y_loss", "c35_x_loss", "c35_y_loss", "c36_x_loss", "c36_y_loss", "c37_x_loss", "c37_y_loss", "c38_x_loss", "c38_y_loss", "c39_x_loss", "c39_y_loss", "c40_x_loss", "c40_y_loss", "c41_x_loss", "c41_y_loss", "c42_x_loss", "c42_y_loss", "c43_x_loss", "c43_y_loss", "c44_x_loss", "c44_y_loss", "c45_x_loss", "c45_y_loss", "c46_x_loss", "c46_y_loss", "c47_x_loss", "c47_y_loss", "c48_x_loss", "c48_y_loss"
        return yolo.corners.CornersValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
