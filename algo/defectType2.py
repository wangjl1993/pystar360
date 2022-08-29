################################################################################
#### 异物类型故障
################################################################################

import cv2
import numpy as np

from pystar360.algo.algoBase import algoBaseABC, algoDecorator
from pystar360.utilities.helper import crop_segmented_rect, frame2rect
from pystar360.ano.inference import PatchCoreInfer


from pystar360.utilities._logger import d_logger


@algoDecorator
class DetectForeignObjectWholeImage(algoBaseABC):
    """检测是否异常，异物/破损都可以使用
    yaml example
    ------------
    xxx:
        module: "pystar360.algo.defectType2"
        func: "DetectForeignObjectWholeImage"
        params:
            memory_bank_path: ".xxxx.ckpt"
            backbone_model_path: ".resnet18.ckpt"
            config_path: "./pystar360/ano/lib1/models/pat/config.yaml"
            image_size: 256
            conf_thres: 0.5
            flip_lr: False # 是否左右翻转
            flip_ud: False # 是否上下翻转
    """

    def __call__(self, item_bboxes_list, test_img, test_startline, img_h, img_w):
        # if empty, return empty
        if not item_bboxes_list:
            return []

        # initialize model
        model = PatchCoreInfer(
            self.item_params["memory_bank_path"],
            self.item_params["backbone_model_path"],
            self.item_params["config_path"],
            self.device,
            image_size=self.item_params["image_size"],
            logger=self.logger,
            mac_password=self.mac_password,
        )
        conf_thres = self.item_params["conf_thres"]

        # iterate
        new_item_bboxes_list = []
        count = 1
        for _, box in enumerate(item_bboxes_list):

            # get proposal rect
            proposal_rect_f = box.proposal_rect
            # convert it to local rect point
            proposal_rect_p = frame2rect(proposal_rect_f, test_startline, img_h, img_w, axis=self.axis)
            img = crop_segmented_rect(test_img, proposal_rect_p)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if self.item_params.get("flip_lr", False):
                img = np.flip(img, axis=1)
            elif self.item_params.get("flip_ud", False):
                img = np.flip(img, axis=0)

            # infer
            _, score = model.infer(img)

            box.index = count
            box.conf_score = score
            box.conf_thres = conf_thres
            # box.curr_rect = box.proposal_rect
            box.is_defect = 1 if score > conf_thres else 0
            box.is_detected = 1
            box.description = f">>> box: {box.name}; score {score}; threshold: {conf_thres}."
            new_item_bboxes_list.append(box)
            count += 1

            if self.debug:
                if self.logger:
                    self.logger.info(box.description)
                else:
                    d_logger.info(box.description)

        return new_item_bboxes_list


@algoDecorator
class DetectForeignObjectAfterSeg(algoBaseABC):
    def __call__(self, item_bboxes_list, test_img, test_startline, img_h, img_w):
        raise NotImplementedError
