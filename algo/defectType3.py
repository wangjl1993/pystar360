################################################################################
#### 破损类故障算法汇总
################################################################################

import cv2
import copy

from pystar360.algo.algoBase import algoBaseABC, algoDecorator
from pystar360.yolo.inference import YoloInfer, yolo_xywh2xyxy_v2
from pystar360.utilities.helper import crop_segmented_rect, frame2rect
from pystar360.utilities._logger import d_logger
from pystar360.base.dataStruct import CarriageInfo, BBox
from typing import List

@algoDecorator
class DetectAtMostNObj(algoBaseABC):
    """检测最多N个obj，如果n=0， 检测出一个目标就算报错
    yaml example
    ------------
    xxx【玻璃裂痕破损】:
        module: "pystar360.algo.defectType3"
        func: "DetectAtMostNObj"
        params:
            model_path: "xxxx.pt"
            imgsz: 640
            conf_thres: 0.45
            iou_thres: 0.2
            at_most_n: 0
    """

    def __call__(self, item_bboxes_list: List[BBox], curr_train2d: CarriageInfo, **kwargs) -> List[BBox]:
        # if empty, return empty
        if not item_bboxes_list:
            return []

        # initialize model
        model = YoloInfer(
            self.item_params["model_path"],
            self.device,
            imgsz=self.item_params["imgsz"],
            logger=self.logger,
            mac_password=self.mac_password,
        )
        at_most_n = self.item_params.get("at_most_n", 0)

        # iterate
        new_item_bboxes_list = []
        count = 1
        for _, box in enumerate(item_bboxes_list):

            # get proposal rect
            proposal_rect_f = box.curr_proposal_rect
            # convert it to local rect point
            proposal_rect_p = frame2rect(proposal_rect_f, curr_train2d.startline, curr_train2d.img_h, curr_train2d.img_w, axis=self.axis)
            img = crop_segmented_rect(curr_train2d.img, proposal_rect_p)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # infer
            outputs = model.infer(
                img, conf_thres=self.item_params["conf_thres"], iou_thres=self.item_params["iou_thres"]
            )

            actual_num = len(outputs)
            if actual_num > at_most_n:
                # 0:class, 1:ctrx, 2:ctry, 3;w, 4:h, 5:confidence,
                max_outputs = max(outputs, key=lambda x: x[5])
                box.conf_score = max_outputs[5]
                box.is_defect = 1  # defect

            # update box info
            box.conf_thres = self.item_params["conf_thres"]
            box.index = count
            box.is_detected = 1
            box.description = f">>> box: {box.name, box.index}; outputs: {actual_num}, defect: {box.is_defect}"
            new_item_bboxes_list.append(box)
            count += 1

            if self.debug:
                if self.logger:
                    self.logger.info(box.description)
                else:
                    d_logger.info(box.description)

        return new_item_bboxes_list
