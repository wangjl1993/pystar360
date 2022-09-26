from PIL import Image
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms

from functools import lru_cache
from typing import Union
from omegaconf import OmegaConf, DictConfig
@lru_cache(maxsize=32, typed=False) # 添加lru缓存机制
class ClsInfer:
    def __init__(self, model_type, model_params, model_path, device, transform_funs_params: Union[dict, DictConfig], logger=None):
        self.model_type = model_type
        self.model_params = model_params
        self.model_path = model_path
        self.device = device
        self.logger = logger
        self.transform_funs_params = transform_funs_params
        self._initialize()
        self._init_transform_funs()

    @torch.no_grad()
    def _initialize(self):
        try:
            model = getattr(models, self.model_type)(**self.model_params)
        except ValueError:
            if self.logger:
                self.logger.error(f"Please provide a valid model name {self.model_type}")
            raise ValueError(f"Please provide a valid model name {self.model_type}")

        with open(self.model_path, "rb")as f:
            checkpoint = torch.load(f, map_location=self.device)
        model.load_state_dict(checkpoint, strict=False)

        model.to(self.device)
        self.model = model.eval()

    def _init_transform_funs(self,):
        if isinstance(self.transform_funs_params, DictConfig):
            self.transform_funs_params = OmegaConf.to_container(self.transform_funs_params)
        transform_funs = []
        for f, params in self.transform_funs_params.items():
            if isinstance(params, dict):
                fun = getattr(transforms, f)(**params)
            elif isinstance(params, list):
                fun = getattr(transforms, f)(*params)
            elif params is None:
                fun = getattr(transforms, f)()
            else:
                if self.logger:
                    self.logger.error(f"transform_funs's params must be in [dict, list, None].")
                raise TypeError("transform_funs's params must be in [dict, list, None].")
            transform_funs.append(fun)
        
        self.transform_funs = transforms.Compose(transform_funs)

    @torch.no_grad()
    def infer_by_cls(self, img: Union[np.ndarray, Image.Image]):
        """inference by a given classfication model."""
        
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")
        
        img = self.transform_funs(img).unsqueeze(0).to(self.device)
        outputs = self.model(img)
        _, predicted = torch.max(outputs.data, 1)

        if self.logger:
            self.logger.debug(outputs.data)

        return predicted.item()

# @torch.no_grad()
# def initialize_cls(model_type, params, path, device):
#     """initialize classification, load paramter hyperparameters, parameters to a given device"""
#     if "mobilenet_v3_small" in model_type:
#         model = mobilenet_v3_small(**params)
#     else:
#         raise ValueError("Please give a correct model name")

#     with open(path, "rb") as f:
#         checkpoint = torch.load(f, map_location=device)
#     model.load_state_dict(checkpoint["state_dict"], strict=False)

#     model.to(device)
#     model = model.eval()
#     return model

#     # import torch
#     # from torchvision.models import mobilenet_v3_small
#     # model = mobilenet_v3_small(num_classes=2)
#     # device = "cuda:1"
#     # path = "./exp1/1/model_best.pth"
#     # with open(path, "rb") as f:
#     #     checkpoint = torch.load(f, map_location=device)
#     # model.load_state_dict(checkpoint["state_dict"], strict=False)
#     # torch.save(model.state_dict(), path)


# def get_img_array(img):
#     """normalize before put data into model"""
#     img = np.float32(img) / 127.5 - 1
#     return img


