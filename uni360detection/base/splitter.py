
from pathlib import Path

import numpy as np
import uni360detection.base.global_settings as SETTINGS
from skimage.metrics import structural_similarity as ssim
from uni360detection.utilities.fileManger import *
from uni360detection.utilities.helper import *
from uni360detection.yolo.inference import YoloInfer, yolo_xywh2xyxy


def find_approximate_single_end(l,
                                var_threshold=1,
                                corr_thres=None,
                                reverse=False,
                                axis=1,
                                imread=imread_tenth):
    """
    找到车头车尾所在大致帧数, 如果没有额外信息可以从图片上调整得出，如果有额外轴信息，可以覆盖
    偏暗使用小一点的var 0.1左右
    偏亮使用大一点的var 1左右
    """
    length = len(l)
    find_end = False

    # backward
    if reverse:
        sidx = length - 1
        eidx = length - int(length // 3)
        step = -1
    else:  # forward
        sidx = 0
        eidx = int(length // 3)
        step = 1

    idx = 0
    for i in range(sidx, eidx, step):
        img = imread(l[i])
        hist = img.mean(axis=0) if axis == 1 else img.mean(axis=1)
        var1 = np.var(hist)
        if var1 > var_threshold:
            if corr_thres is not None:
                corr = ssim(img, imread(l[i + step]))
                if corr < corr_thres:
                    idx = i
                    find_end = True
                    break
            else:
                idx = i
                find_end = True
                break

    if not find_end:
        raise ValueError("Couldn't find end.")
    return idx


def select_best_cutpoints(candidates, method):
     # 0:class, 1:ctrx, 2:ctry, 3;w, 4:h, 5:confidence, 6:startline, 7:endline
    if method == "width":
        max_candidate = max(candidates, key=lambda x: x[3]) # max width
    elif method == "height":
        max_candidate = max(candidates, key=lambda x: x[4]) # max height
    elif method == "area":
        max_candidate = max(candidates, key=lambda x: x[4] * x[3]) # max area
    elif method == "confidence":
        max_candidate = max(candidates, key=lambda x: x[5]) # max confidence
    else:
        raise NotImplementedError(f">>> method {method} is not implemented")
    return max_candidate

class Splitter:
    def __init__(self,
                 qtrain_info,
                 local_params,
                 images_path_list,
                 train_library_path,
                 device,
                 logger=None):
        
        # query information 
        self.qtrain_info = qtrain_info 
        self.major_train_code = qtrain_info.major_train_code
        self.minor_train_code = qtrain_info.minor_train_code
        self.train_num = qtrain_info.train_num
        self.train_sn = qtrain_info.train_sn
        self.channel = qtrain_info.channel
        self.carriage = qtrain_info.carriage
        
        # local params 
        self.params = local_params
        self.axis = local_params.axis
        self.var_threshold = local_params.splitter.get("var_threshold", 1)

        # images list 
        self.images_path_list = images_path_list

        train_library_path = Path(train_library_path) / (
            qtrain_info.major_train_code + ".yaml")
        self.train_dict = read_yaml(str(
            train_library_path))[self.minor_train_code]

        self.device = device
        self.logger = logger

        self._cutframe_idx = None
    
    def get_approximate_cutframe_idxes(self):
        head_appro_idx = find_approximate_single_end(
            self.images_path_list,
            var_threshold=self.var_threshold,
            axis=self.axis)
        tail_appro_idx = find_approximate_single_end(
            self.images_path_list,
            var_threshold=self.var_threshold,
            reverse=True,
            axis=self.axis)
        n_frames = tail_appro_idx - head_appro_idx + 1
        default_cutpoints = np.array(self.train_dict.cutpoints)
        cutframe_idxes = np.rint(n_frames * default_cutpoints + head_appro_idx)

        # cutframe_idxes[self.car-1], cutframe_idxes[self.car]
        return cutframe_idxes

    @property
    def cutframe_idx(self):
        return self._cutframe_idx

    @cutframe_idx.setter
    def cutframe_idx(self, cutframe_idx):
        self._cutframe_idx = cutframe_idx

    def update_cutframe_idx(self, *cutframe_idx):
        assert len(cutframe_idx) == 2
        self.cutframe_idx = cutframe_idx

    def get_specific_cutpoints(self,
                               cover_range=2,
                               shift=3,
                               offset=0, 
                               imread=imread_quarter,
                               save_path=None):
        if self.cutframe_idx is None:
            raise ValueError("Please provide cutframe index 轴信息.")
        assert len(self.cutframe_idx) == 2

        # load model
        model = YoloInfer(self.params.splitter.model_path,
                          self.device,
                          self.params.splitter.imgsz,
                          logger=self.logger)
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

        _cutpoints = []
        for p, cutframe_idx in enumerate(self.cutframe_idx):
            possible_outputs = []
            if p == 0:
                cutframe_idx -= offset 
            else:
                cutframe_idx += offset 

            for i in range(shift):
                index = cutframe_idx + i - (shift // 2)
                startline = int(max(0, index - cover_range))
                endline = int(
                    min(len(self.images_path_list), index + cover_range + 1))
                img = read_segmented_img(self.images_path_list, startline,
                                         endline, imread, axis=self.axis)
                if save_path:
                    fname = save_path / f"{self.train_sn}_{self.minor_train_code}_{self.channel}_{self.car}_{p}_{i}.jpg"
                    cv2.imwrite(str(fname), img)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                temp_outputs = model.infer(img, conf_thres=self.params.splitter.conf_thres, iou_thres=self.params.splitter.iou_thres)
                for out in temp_outputs:
                     # 0:class, 1:ctrx, 2:ctry, 3;w, 4:h, 5:confidence, 6:startline, 7:endline
                    out = list(out)
                    possible_outputs.append(out+[startline, endline])

            new_cutpoints = self._post_process(possible_outputs, p)
            _cutpoints.append(new_cutpoints)
                
        return _cutpoints

    def _post_process(self, outputs, p):
        # return specific cutpoints
        if (self.carriage == 1 and p == 0 ) or (self.carriage == self.train_dict.num and p == 1):
            outputs = [i for i in outputs if self.params.splitter.label_translator[int(i[0])] == "end"]
        else:
            outputs = [i for i in outputs if self.params.splitter.label_translator[int(i[0])] == "mid"]

        if len(outputs) < 1:
            raise ValueError("Can't find cut line for splitting carriage.")
        
        max_output = select_best_cutpoints(outputs, self.params.splitter.method)
        startline = max_output[6] #6: startline
        endline = max_output[7] #7: endline 
        if self.carriage == 1 or self.carriage == self.train_dict.num: 
            # first carraige or last carriage 
            if self.axis == 1:
                # x axis cutline 
                point = yolo_xywh2xyxy(max_output[1:5], startline, 0, 1., endline-startline)
                if p == 0:
                    if self.carriage == self.train_dict.num:
                        return point[1][0]
                    else:
                        return point[0][0]
                elif p == 1:
                    return point[1][0]
            elif self.axis == 0:
                # y axis cutline
                point = yolo_xywh2xyxy(max_output[1:5], 0, startline, endline-startline, 1.)
                if p == 0:
                    if self.carriage == self.train_dict.num:
                        return point[1][1]
                    else:
                        return point[0][1]
                elif p == 1:
                    return point[1][1]
            else:
                raise ValueError(f"Axis {self.axis} is not available.")
        else:
            # middle carriages
            if self.axis == 1:
                point = yolo_xywh2xyxy(max_output[1:5], startline, 0, 1., endline-startline)
                return point[1][0]
            elif self.axis == 0:
                point = yolo_xywh2xyxy(max_output[1:5], 0, startline, endline-startline, 1.)
                return point[1][1]
            else:
                raise ValueError(f"Axis {self.axis} is not available.")

    def _generate_cutpoints_img(self, save_path, imread=imread_quarter, aux="", cover_range=2, shift=3, offset=0):
        if self.cutframe_idx is None:
            raise ValueError("Please provide cutframe index 轴信息.")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        for p, cutframe_idx in enumerate(self.cutframe_idx):
            if p == 0:
                cutframe_idx -= offset 
            else:
                cutframe_idx += offset 
            for i in range(shift):
                index = cutframe_idx + i - (shift // 2)
                startline = int(max(0, index - cover_range))
                endline = int(
                    min(len(self.images_path_list), index + cover_range + 1))
                img = read_segmented_img(self.images_path_list, startline,
                                         endline, imread, axis=self.axis)
                fname = save_path / f"{aux}_{self.minor_train_code}_{self.train_num}_{self.train_sn}_{self.channel}_{self.carriage}_{p}_{i}.jpg"
                cv2.imwrite(str(fname), img)
                print(f">>> {fname}.")
