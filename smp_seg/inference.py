import segmentation_models_pytorch as smp
import torch
import cv2
from pathlib import Path
from omegaconf import DictConfig
from functools import lru_cache
import numpy as np
import albumentations as albu
from pystar360.base.baseInferencer import BaseInfer
from pystar360.utilities.de import decrpt_content_from_filepath
from pystar360.utilities.logger import w_logger

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

@lru_cache(maxsize=8, typed=False)  # 添加lru缓存机制，参数对象必须可hash
class SmpInfer(BaseInfer):
    def __init__(self, model_type: str, model_params: DictConfig, model_path: str, device: str, mac_password=None):
        self.model_type = model_type
        self.model_params = model_params
        self.model_path = model_path
        self.device = device
        self.mac_password = mac_password
        self._initialize()

    def _initialize(self):
        try:
            model = getattr(smp, self.model_type)(**self.model_params, encoder_weights=None) # set encoder_weights=None, otherwise smp will download imagenet pretrained model.
        except ValueError:
            w_logger.error(f"Please provide a valid model name {self.model_type} or model params {self.model_params}")
            raise ValueError(f"Please provide a valid model name {self.model_type} or {self.model_params}, please see https://github.com/qubvel/segmentation_models.pytorch")

        if self.mac_password:
            self.model_path = Path(self.model_path)
            self.model_path = self.model_path.with_suffix(".pystar")
        
        checkpoint = torch.load(
            f=decrpt_content_from_filepath(self.model_path, self.mac_password), 
            map_location=self.device
        )
        model.load_state_dict(checkpoint, strict=False)
        model.to(self.device)
        self.model = model.eval()

    def get_transform_funs(self, H: int, W: int, pretrained: str="imagenet"):
        
        preprocessing_fun = smp.encoders.get_preprocessing_fn(self.model_params["encoder_name"], pretrained)
        transform_funs = albu.Compose([
            albu.Resize(H, W),
            albu.Lambda(image=preprocessing_fun),
            albu.Lambda(image=to_tensor)
        ])
        self.transform_funs = transform_funs


    def infer(self, img: np.ndarray, thres: float=0.5):
        """inference by a given single-class smp_model."""

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        H, W = img.shape[:2]
        input = self.transform_funs(image=img)["image"]
        x_tensor = torch.from_numpy(input).to(self.device).unsqueeze(0)
        pre_mask = self.model.predict(x_tensor)
        pre_mask = (pre_mask.squeeze().cpu().numpy()>thres).astype(np.uint8)
        pre_mask = cv2.resize(pre_mask, (W,H))
        return pre_mask
