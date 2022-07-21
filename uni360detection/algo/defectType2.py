################################################################################
#### 异物类型故障
################################################################################

import cv2 
import copy
import numpy as np

from uni360detection.algo.algoBase import algoBase, algoDecorator
from uni360detection.utilities.helper import crop_segmented_rect, frame2rect
from uni360detection.ano.inference import PatchCoreInfer


@algoDecorator
class DetectForeignObjectWholeImage(algoBase):
    def __call__(self, item_bboxes_list, test_img, test_startline, img_h, img_w):
        # if empty, return empty 
        if not item_bboxes_list:
            return []

        # initialize model 
        model = PatchCoreInfer(self.item_params["memory_bank_path"],
                               self.item_params["backbone_model_path"],
                               self.item_params["config_path"],
                               self.device,
                               image_size=self.item_params["image_size"],
                               logger=self.logger)
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
            box.curr_rect = proposal_rect_f
            box.is_defect = 1 if score > conf_thres else 0
            box.description = f">>> box: {box.name}; score {score}; threshold: {conf_thres}."
            new_item_bboxes_list.append(box)
            count += 1 

            if self.logger:
                self.logger.info(box.description)
            else:
                print(box.description) 
        
        return new_item_bboxes_list



@algoDecorator
class DetectForeignObjectAfterSeg(algoBase):
    def __call__(self, item_bboxes_list, test_img, test_startline, img_h, img_w):
        raise NotImplementedError


