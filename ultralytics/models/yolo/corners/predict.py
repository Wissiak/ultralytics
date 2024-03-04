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
        # preds[0] is the inference result
        # preds[0][:,:15] is the bounding box
        # preds[0][:,15:] = preds[1][1] is the corners
        bboxes, p_masks = ops.non_max_suppression(
            preds[0][:,:15],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            need_mask=True
        )

        pred_corners = preds[1][1]
        pred_corners = pred_corners.permute(0, 2, 1)
        #nms_corners = [None] * max(mask)
        nms_corners = []
        for pred_i, mask_i in enumerate(p_masks):
            nms_corners.append(pred_corners[pred_i][mask_i])

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, bbox in enumerate(bboxes):
            orig_img = orig_imgs[i]
            bbox[:, :4] = ops.scale_boxes(img.shape[2:], bbox[:, :4], orig_img.shape) # xy, xy
            img_path = self.batch[0][i]
            corners = nms_corners[i].reshape(-1, 12, 2)
            corners = ops.scale_corners(corners, bbox[:,2:4] - bbox[:,:2])

            corners += bbox[:,:2].view(bbox[:,:2].shape[0], 1, bbox[:,:2].shape[1]).broadcast_to(corners.shape) # make relative to bounding box
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=bbox, corners=corners))
        return results
