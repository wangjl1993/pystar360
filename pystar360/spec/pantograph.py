
import cv2
from dataclasses import dataclass

from pystar360.utilities.fileManger import *
from pystar360.utilities.helper import *
from pystar360.yolo.inference import YoloInfer, yolo_xywh2xyxy_v2


@dataclass
class QPantoInfo:
    major_train_code: str = "" # CRH1A
    minor_train_code: str = "" # CRH1A-A
    train_num: str= "" # 1178, CRH1A-A 1178
    train_sn: str = "" # 2101300005, date or uid 
    channel: str = "" # 12,4,17...
    path: str = ""

def select_best_panto(candidates, method):
     # 0:class, 1:ctrx, 2:ctry, 3;w, 4:h, 5:confidence, 6:index
    if method == "width":
        max_candidate = max(candidates, key=lambda x: x[3]) # max width
    elif method == "height":
        max_candidate = max(candidates, key=lambda x: x[4]) # max height
    elif method == "area":
        max_candidate = max(candidates, key=lambda x: x[4] * x[3]) # max area
    elif method == "confidence":
        max_candidate = max(candidates, key=lambda x: x[5]) # max confidence
    elif method == "maxy":
        max_candidate = max(candidates, key=lambda x: x[2]) # max confidence
    else:
        raise NotImplementedError(f">>> method {method} is not implemented")
    return max_candidate

class DetectPantograph:
    def __init__(self, query_panto_info,
                     channel_params, device, logger=None):
        self.query_panto_info = query_panto_info
        self.channel_params = read_yaml(channel_params)[query_panto_info.channel]
        self.device = device 
        self.logger = logger 
        self.__post_init()


    def __post_init(self):
        filter_rules = []
        filter_rule1 = lambda l: [p for p in l if p.stem.split("-")[0] == self.query_panto_info.channel]
        filter_rules.append(filter_rule1)
        self.img_path_list = list_images_in_path(self.query_panto_info.path, filter_rules=filter_rules)
        self.img_length = len(self.img_path_list)


    def locate_panto(self):
        locator_params = self.channel_params.locator 

        if self.img_length < 1:
            return 

        model = YoloInfer(locator_params.model_path, self.device, imgsz=locator_params.imgsz)
        is_panto_found = False 
        for r in locator_params.ratio_location:
            idx = int(r * self.img_length)
            sidx = max(idx - locator_params.cover_range, 0)
            eidx = min(idx + locator_params.cover_range + 1, self.img_length)
            
            if self.logger:
                self.logger(f">>> check from {self.img_path_list[sidx]} to {self.img_path_list[eidx]}")
            else:
                print(f">>> check from {self.img_path_list[sidx]} to {self.img_path_list[eidx]}")

            possible_outputs = []
            for img_idx in range(sidx, eidx):
                img = imread_full(self.img_path_list[img_idx])
                img_h, img_w = img.shape
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                outputs = model.infer(img, conf_thres=locator_params.conf_thres, iou_thres=locator_params.iou_thres)
                # 0:class, 1:ctrx, 2:ctry, 3;w, 4:h, 5:confidence, 6:index, 7:img_h, 8:img_w
                outputs = [list(i) + [img_idx, img_h, img_w] for i in outputs if locator_params.label_translator[int(i[0])] == locator_params.target_label]

                possible_outputs.extend(outputs)
            
            if len(possible_outputs) > 0:
                max_output = select_best_panto(possible_outputs, locator_params.method)
                is_panto_found = True
                break 
        
        if is_panto_found:
            img_loc_idx = max_output[-3]
            img_h, img_w = max_output[7:9]
            rect = yolo_xywh2xyxy_v2(max_output[1:5], [[0,0],[img_w, img_h]])
            rect = [list(map(int, r)) for r in rect]
            return img_loc_idx, rect 
        else:
            # not found 
            return

    # def segment_panto(self, img_loc_idx, rect):

    #     img = imread_full(self.img_path_list[img_loc_idx])
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     img = crop_segmented_rect(img, rect)











                



