################################################################################
#### 丢失类故障算法汇总
################################################################################

import cv2
import copy
import numpy as np
from pystar360.algo.algoBase import algoBaseABC, algoDecorator
from pystar360.yolo.inference import YoloInfer, yolo_xywh2xyxy_v2
from pystar360.utilities.helper import crop_segmented_rect, frame2rect, hungary_match
from pystar360.utilities.helper3d import map3d_for_bboxes

from pystar360.utilities._logger import d_logger
from pystar360.base.dataStruct import CarriageInfo, BBox
from typing import List, Optional, Callable
@algoDecorator
class DetectItemsMissing(algoBaseABC):
    """检测是否丢失

    yaml example
    ------------
    xxxx:
    module: "pystar360.algo.defectType1"
    func: "DetectItemsMissing"
    params:
        model_path: "xxxx.pt"
        imgsz: 640
        conf_thres: 0.45
        iou_thres: 0.2
        label_translator: {0: 'xxxx'}
    """

    def __call__(
            self, item_bboxes_list: List[BBox], curr_train2d: CarriageInfo, curr_train3d: Optional[CarriageInfo]=None, 
            hist_train2d: Optional[CarriageInfo]=None, hist_train3d: Optional[CarriageInfo]=None,
            map3d_minor_axis_poly_func: Optional[Callable]=None, map3d_major_axis_poly_func: Optional[Callable]=None, **kwargs
        ) -> List[BBox]:
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

        # iterate
        new_item_bboxes_list = []
        count = 1
        for _, box in enumerate(item_bboxes_list):
            # get update box info
            label_name = box.name
            num2check = box.num2check

            # get proposal rect
            curr_proposal_rect_f = box.curr_proposal_rect
            # convert it to local rect point
            curr_proposal_rect_p = frame2rect(curr_proposal_rect_f, curr_train2d.startline, curr_train2d.img_h, curr_train2d.img_w, axis=self.axis)
            curr_img = crop_segmented_rect(curr_train2d.img, curr_proposal_rect_p)
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)

            # infer
            curr_outputs = model.infer(curr_img, conf_thres=self.item_params["conf_thres"], iou_thres=self.item_params["iou_thres"])
            curr_outputs = [i for i in curr_outputs if self.item_params["label_translator"][int(i[0])] == label_name]
            curr_outputs = sorted(curr_outputs, key=lambda x: x[-1])

            # update box info
            curr_actual_num = len(curr_outputs)
            box.conf_thres = self.item_params["conf_thres"]
            box.is_detected = 1  # 是否检查过
            only_check_num = self.item_params.get("item_check_num", False)

            if hist_train2d is not None and not box.hist_proposal_rect.is_none():
                # 
                hist_proposal_rect_f = box.hist_proposal_rect
                hist_proposal_rect_p = frame2rect(hist_proposal_rect_f, hist_train2d.startline, hist_train2d.img_h, hist_train2d.img_w, axis=self.axis)
                hist_img = crop_segmented_rect(hist_train2d.img, hist_proposal_rect_p)
                hist_img = cv2.cvtColor(hist_img, cv2.COLOR_GRAY2BGR)
                hist_outputs = model.infer(hist_img, conf_thres=self.item_params["conf_thres"], iou_thres=self.item_params["iou_thres"])
                hist_outputs = [i for i in hist_outputs if self.item_params["label_translator"][int(i[0])] == label_name]
                hist_outputs = sorted(hist_outputs, key=lambda x: x[-1])
                hist_actual_num = len(hist_outputs)
                box.description = f">>> box: {box.name}; curr actual num: {curr_actual_num}; hist actual num: {hist_actual_num}; num required: {num2check}."

                if curr_actual_num == 0:
                    box.index = count
                    box.curr_rect = curr_proposal_rect_f
                    if hist_actual_num > 0:
                        box.is_defect = 1
                    if self.logger:
                        self.logger.info(box.description)
                    else:
                        d_logger.info(box.description)
                    new_item_bboxes_list.append(box)
                    count += 1
                    continue
                
                if only_check_num:
                    # only check num，do not need to return every single detected item's location
                    box.index = count
                    box.curr_rect, box.hist_rect = curr_proposal_rect_f, hist_proposal_rect_f
                    box.conf_score = curr_outputs[0][-1]  # get the max conf
                    if curr_actual_num < hist_actual_num:
                        box.is_defect = 1  # defect
                    new_item_bboxes_list.append(box)
                    count += 1
                else:
                    curr_outputs_center = np.array([i[1:3] for i in curr_outputs])
                    hist_outputs_center = np.array([i[1:3] for i in hist_outputs])
                    match_res = hungary_match(curr_outputs_center, hist_outputs_center)

                    # 优先收集能和历史图匹配上的点
                    collect_order_list = list(match_res.keys()) + list( set(range(curr_actual_num))-set(match_res.keys()) )[::-1]
                    for i, curr_index in enumerate(collect_order_list, start=1):
                        if i > num2check:
                            break
                        hist_index = match_res[curr_index]
                        new_box = copy.deepcopy(box)
                        new_box.index = count
                        new_box.curr_rect = yolo_xywh2xyxy_v2(curr_outputs[curr_index][1:5], curr_proposal_rect_f) # yolo output: (class_id,x,y,w,h,conf)
                        new_box.hist_rect = yolo_xywh2xyxy_v2(hist_outputs[hist_index][1:5], hist_proposal_rect_f)
                        new_box.conf_score = curr_outputs[curr_index][-1]
                        new_item_bboxes_list.append(new_box)
                        count += 1
                    
                    if curr_actual_num < num2check and curr_actual_num < hist_actual_num:
                        box.index = count
                        box.curr_rect, box.hist_rect = curr_proposal_rect_f, hist_proposal_rect_f
                        box.is_defect = 1
                        new_item_bboxes_list.append(box)
                        count += 1


            else:
                box.description = f">>> box: {box.name}; curr actual num: {curr_actual_num}; num required: {num2check}."
                # if len(outpus) == 0
                if curr_actual_num == 0:
                    box.index = count
                    box.curr_rect = curr_proposal_rect_f
                    box.is_defect = 1
                    if self.logger:
                        self.logger.info(box.description)
                    else:
                        d_logger.info(box.description)
                    # update list
                    new_item_bboxes_list.append(box)
                    count += 1
                    continue
                
                if only_check_num:
                    # only check num，do not need to return every single detected item's location
                    box.index = count
                    box.curr_rect = curr_proposal_rect_f
                    box.conf_score = curr_outputs[0][-1]  # get the max conf
                    if curr_actual_num < num2check:
                        box.is_defect = 1  # defect

                    # update list
                    new_item_bboxes_list.append(box)
                    count += 1
                else:
                    # need to return every single detected item's location
                    iternum = min(curr_actual_num, num2check)
                    for i in range(iternum):
                        # update non-defect box
                        new_box = copy.deepcopy(box)
                        new_box.index = count
                        new_box.curr_rect = yolo_xywh2xyxy_v2(curr_outputs[i][1:5], curr_proposal_rect_f)
                        new_box.conf_score = curr_outputs[i][-1]

                        # update list
                        new_item_bboxes_list.append(new_box)
                        count += 1

                    if curr_actual_num < num2check:
                        # curr actual num < num2check, add one more defect box; else pass
                        # update defect box
                        box.index = count
                        box.curr_rect = curr_proposal_rect_f
                        box.is_defect = 1

                        # update list
                        new_item_bboxes_list.append(box)
                        count += 1
                    else:
                        pass
            if curr_train3d is not None or hist_train3d is not None:
                map3d_for_bboxes(new_item_bboxes_list, self.axis, map3d_minor_axis_poly_func, map3d_major_axis_poly_func)

            if self.debug:
                if self.logger:
                    self.logger.info(box.description)
                else:
                    d_logger.info(box.description)

        return new_item_bboxes_list
