import torch
import numpy as np

from models.experimental import attempt_load
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
)
from utils.augmentations import letterbox
from utils.torch_utils import time_synchronized

def convert_img(img0, imgsz, stride):
    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return img


class YoloInfer:
    def __init__(self, model_path, device, imgsz=640):
        self.model_path = model_path 
        self.device = device
        self.imgsz = imgsz  
        self._initialize()
    
    @torch.no_grad()
    def _initialize(self):
        self.model = attempt_load(self.model_path, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Run inference
        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                    next(self.model.parameters())))  # run once
    
    @torch.no_grad()
    def infer():
        
    

    # def _initialize(self):
    #      """Initialize yolo model"""
    #     self.model = attempt_load(self.model_path, map_location=self.device)  # load FP32 model
    #     self.stride = int(model.stride.max())  # model stride
    #     self.imgsz = check_img_size(imgsz, s=stride)  # check image size



# @torch.no_grad()
# def initialize_yolo(
#         model_path,
#         device,
#         imgsz=640  # inference size (pixels),
# ):
#     """Initialize yolo model"""
#     model = attempt_load(model_path, map_location=device)  # load FP32 model
#     stride = int(model.stride.max())  # model stride
#     imgsz = check_img_size(imgsz, s=stride)  # check image size

#     # Run inference
#     if device.type != "cpu":
#         model(
#             torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
#                 next(model.parameters())))  # run once

#     return model, imgsz, stride



