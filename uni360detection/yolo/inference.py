from weakref import ref
import torch
import numpy as np

from models.experimental import attempt_load
from yolo.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
)
from yolo.utils.augmentations import letterbox
from yolo.utils.torch_utils import time_synchronized

def yolo_xywh2xyxy(points, sx, sy, ref_h, ref_w):

    assert len(points) == 4 , "length is 4, ctrx, ctry, w, h"
    x, y, w, h = points 

    half_h = h / 2.
    half_w = w / 2.

    x_left = sx + (x - half_w) * ref_w
    y_top = sy + (y - half_h) * ref_h
    x_right = sx + (x + half_w) * ref_w
    y_bottom = sy + (y + half_h) * ref_h

    return [(x_left, y_top), (x_right, y_bottom)]

def convert_img(img0, imgsz, stride):
    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return img


class YoloInfer:
    def __init__(self, model_path, device, imgsz=640, logger=None):
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.logger = logger
        self._initialize()

    @torch.no_grad()
    def _initialize(self):
        self.model = attempt_load(self.model_path,
                                  map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz,
                                    s=self.stride)  # check image size

        # Run inference
        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, self.imgsz,
                            self.imgsz).to(self.device).type_as(
                                next(self.model.parameters())))  # run once

    @torch.no_grad()
    def infer(
        self,
        img0,
        conf_thres=0.45,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,
        **kwargs):  # class-agnostic NMS

        img = convert_img(img0, self.imgsz, self.stride)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=False, visualize=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred,
                                   conf_thres,
                                   iou_thres,
                                   classes,
                                   agnostic_nms,
                                   max_det=max_det)
        t2 = time_synchronized()

        # Process detections
        output = []
        for i, det in enumerate(pred):  # detections per image
            s, im0 = "", img0.copy()

            gn = torch.tensor(im0.shape)[[1, 0, 1,
                                          0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = ((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                             gn).view(-1).tolist())  # normalized xywh
                    line = (int(cls.item()), *xywh, conf.item())
                    output.append(line)

        # Print time (inference + NMS)
        if self.logger:
            self.logger.info(f"{s}Done. ({t2 - t1:.3f}s)")

        return output
