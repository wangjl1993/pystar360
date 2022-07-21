import cv2
import numpy as np
import torch 

import warnings 
warnings.filterwarnings("ignore")

from importlib import import_module
from pathlib import Path

from uni360detection.ano.lib1.post_processing import compute_mask
from uni360detection.ano.lib1.config import get_configurable_parameters
from uni360detection.ano.lib1.deploy.inferencers.base import Inferencer
    
class PatchCoreInfer:
    def __init__(self, memory_bank_path, backbone_model_path,
                config_path, device, image_size = 256, logger=None):
        self.memory_bank_path = memory_bank_path
        self.backbone_model_path = backbone_model_path
        self.config_path = config_path 
        self.device = device 
        self.image_size = image_size
        self.logger = logger 
        self._initialize()

    @torch.no_grad()
    def _initialize(self):

        config = get_configurable_parameters(config_path=self.config_path, 
                    image_size=self.image_size)
    
        # Get the inferencer
        extension = Path(self.memory_bank_path).suffix
        self.model: Inferencer
        if extension in (".ckpt", ".pth"):
            module = import_module("uni360detection.ano.lib1.deploy.inferencers.torch")
            TorchInferencer = getattr(module, "TorchInferencer")  # pylint: disable=invalid-name
            self.model = TorchInferencer(config=config, device=self.device, model_source=self.memory_bank_path,
            model_backbone_path=self.backbone_model_path, meta_data_path=None)
        else:
            raise ValueError(
                f"Model extension is not supported. Torch Inferencer exptects a .ckpt file."
            )

    @torch.no_grad()
    def infer(self, image, thres=0.5, return_mask=False):
        anomaly_map, score = self.model.predict(image=image)

        if return_mask:
            mask = compute_mask(anomaly_map, thres) # assumes predictions are normalized.
            return anomaly_map, score, mask

        return anomaly_map, score
        

def find_countour(thresh_img):
    cnts = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts_rect = [cv2.boundingRect(c) for c in cnts]
    return cnts_rect

def cal_hist(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])

def cal_hist_density(hist, diff=1):
    return hist / (sum(hist) * diff)

def classify_patchcore(image, rects, threshold, cnt_threshold=2, resolution=50):
    ori_img_hist = cal_hist(image)
    ori_img_density = cal_hist_density(ori_img_hist).ravel()
    
    cls_res = np.zeros(len(rects))
    for idx, rect in enumerate(rects):
        x, y, w, h = rect
        step = int(h/resolution)
        if step < 3:
            rect_img = image[y:y+h, x:x+w].copy()
            rect_img_hist = cal_hist(rect_img)
            rect_img_density = cal_hist_density(rect_img_hist).ravel()
            cor = np.corrcoef(ori_img_density, rect_img_density)[0][1]
            
            if cor <= threshold:
                cls_res[idx] = 1 
        else:
            cnt = 0
            for i in range(1, step):
                rect_img = image[y+(resolution*(i-1)):y+resolution*i, x:x+w].copy()
                rect_img_hist = cal_hist(rect_img)
                rect_img_density = cal_hist_density(rect_img_hist).ravel()
                cor = np.corrcoef(ori_img_density, rect_img_density)[0][1]
                if cor <= threshold:
                    cnt += 1
        
            if cnt >= cnt_threshold:
                cls_res[idx] = 1 
        
    return cls_res

