
################################################################################
#### 丢失类故障算法汇总
################################################################################

import cv2 
import copy

from pystar360.algo.algoBase import algoBaseABC, algoDecorator
from pystar360.yolo.inference import YoloInfer, yolo_xywh2xyxy_v2
from pystar360.utilities.helper import crop_segmented_rect, frame2rect


@algoDecorator
class DetectItemsMissing(algoBaseABC):
    def __call__(self, item_bboxes_list, test_img, test_startline, img_h, img_w):
        # if empty, return empty 
        if not item_bboxes_list:
            return []
        
        # initialize model 
        model = YoloInfer(self.item_params["model_path"], self.device, 
                        imgsz=self.item_params["imgsz"], logger=self.logger, 
                        mac_password=self.mac_password)
        
        # iterate 
        new_item_bboxes_list = []
        count = 1
        for _, box in enumerate(item_bboxes_list):
            # get update box info
            label_name = box.name 
            num2check = box.num2check 

            # get proposal rect 
            proposal_rect_f = box.proposal_rect 
            # convert it to local rect point 
            proposal_rect_p = frame2rect(proposal_rect_f, test_startline, img_h, img_w, axis=self.axis)
            img = crop_segmented_rect(test_img, proposal_rect_p)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # infer
            outputs = model.infer(img, conf_thres=self.item_params["conf_thres"], iou_thres=self.item_params["iou_thres"])
            outputs = [i for i in outputs if self.item_params["label_translator"][int(i[0])] == label_name]

            # update box info 
            actual_num = len(outputs)
            box.conf_thres = self.item_params["conf_thres"]
            box.description = f">>> box: {box.name}; actual num: {actual_num}; num required: {num2check}."
            box.is_detected = 1 # 是否检查过

            # if len(outpus) == 0
            if  actual_num == 0 and num2check > 0:
                box.index = count
                box.curr_rect = proposal_rect_f 
                box.is_defect = 1
                
                if self.logger:
                    self.logger.info(box.description)
                else:
                    print(box.description) 
                # update list 
                new_item_bboxes_list.append(box)
                count += 1
                continue 

            only_check_num = self.item_params.get("item_check_num", False)
            outputs = sorted(outputs, key=lambda x: x[0])
            if only_check_num:
                # only check num，do not need to return every single detected item's location 
                box.index = count
                box.curr_rect = proposal_rect_f 
                box.conf_score = outputs[0][0] # get the max conf
                if actual_num < num2check:
                    box.is_defect = 1 # defect 
                
                # update list 
                new_item_bboxes_list.append(box)
                count += 1
            else:
                # need to return every single detected item's location
                iternum = min(actual_num, num2check)
                for i in range(iternum):
                    # update non-defect box 
                    new_box = copy.deepcopy(box)
                    new_box.index = count
                    new_box.curr_rect = yolo_xywh2xyxy_v2(outputs[i][1:5], proposal_rect_f)
                    new_box.conf_score = outputs[i][0] 

                    # update list 
                    new_item_bboxes_list.append(new_box)
                    count += 1

                if actual_num < num2check:
                    # actual num < num2check, add one more defect box; else pass 
                    # update defect box 
                    box.index = count
                    box.curr_rect = proposal_rect_f 
                    box.is_defect = 1

                    # update list 
                    new_item_bboxes_list.append(box)
                    count += 1
                else:
                    pass

            if self.logger:
                self.logger.info(box.description)
            else:
                print(box.description) 

        return new_item_bboxes_list


