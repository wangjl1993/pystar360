import cv2
import numpy as np
import torch
import torchvision.models as models


class ClsInfer:
    def __init__(self, model_type, model_params, model_path, device, logger=None):
        self.model_type = model_type
        self.model_params = model_params
        self.model_path = model_path
        self.device = device
        self.logger = logger
        self._initialize()

    @torch.no_grad()
    def _initialize(self):
        try:
            model = getattr(models, self.model_type)(**self.model_params)
        except ValueError:
            if self.logger:
                self.logger.error(f"Please provide a a valid model name {self.model_type}")
            raise ValueError(f"Please provide a a valid model name {self.model_type}")

        with open(self.model_path, "rb")as f:
            checkpoint = torch.load(f, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        model.to(self.device)
        self.model = model.eval()

    @torch.no_grad()
    def infer_by_cls(self, img):
        """inference by a given classfication model"""
        # img = cv2.resize(img, (img_size, img_size))
        # img = get_img_array(img)
        # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # img = np.ascontiguousarray(img)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).to(self.device)
        else:
            img = img.to(self.device)
        img = img.float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
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


