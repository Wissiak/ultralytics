# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import numpy as np
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
        bboxes = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            nc=len(self.model.names),
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        pred_corners = preds[1][1]
        pred_corners = pred_corners.permute(0, 2, 1)
        #nms_corners = [None] * max(mask)
        #nms_corners = []
        #for pred_i, mask_i in enumerate(p_masks):
        #    nms_corners.append(pred_corners[pred_i][mask_i])

        # import cv2
        # width, height = (bboxes[0][:, 2:4] - bboxes[0][:, :2])[0]
        # for c in bboxes[0][:,-48:]:
        #     img_ = img[0].cpu().numpy().transpose(1, 2, 0)
        #     img_ = img_.copy()
        #     corners = c.cpu().numpy().reshape(-1, 2) * [width, height]
          
        #     for corner in corners:
        #         cv2.circle(img_, (int(corner[0]), int(corner[1])), 5, (0, 0, 255), -1)
        #     cv2.imshow('corners', img_)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, bbox in enumerate(bboxes):
            orig_img = orig_imgs[i]
            bbox[:, :4] = ops.scale_boxes(img.shape[2:], bbox[:, :4], orig_img.shape) # xy, xy
            img_path = self.batch[0][i]
            corners = bbox[:,-48:].reshape(-1, 24, 2)
            corners = ops.scale_corners(corners, 640)

            #corners += bbox[:,:2].view(bbox[:,:2].shape[0], 1, bbox[:,:2].shape[1]).broadcast_to(corners.shape) # make relative to bounding box
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=bbox[:,:6], corners=corners))
        return results
