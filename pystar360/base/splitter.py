
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity as ssim
from pystar360.utilities.fileManger import *
from pystar360.utilities.helper import *
from pystar360.yolo.inference import YoloInfer, yolo_xywh2xyxy, select_best_yolobox

EPS = 1e-6

def find_approximate_single_end(l,
                                var_threshold=1,
                                max_var_threshold=200,
                                corr_thres=None,
                                reverse=False,
                                axis=1,
                                skip_num=0,
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
        sidx = length - 1 - skip_num
        eidx = length - int(length // 3)
        step = -1
    else:  # forward
        sidx = 0 + skip_num
        eidx = int(length // 3)
        step = 1

    idx = 0
    for i in range(sidx, eidx, step):
        img = imread(l[i])
        hist = img.mean(axis=0) if axis == 1 else img.mean(axis=1)
        var1 = np.var(hist)
        if max_var_threshold > var1 > var_threshold:
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
    

class Splitter:
    def __init__(self,
                 qtrain_info,
                 local_params,
                 images_path_list,
                 train_library_path,
                 device,
                 axis=1,
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

        # images list 
        self.images_path_list = images_path_list

        train_library_path = Path(train_library_path) / (qtrain_info.major_train_code + ".yaml")
        self.train_dict = read_yaml(str(train_library_path))[self.minor_train_code]

        self.device = device
        self.axis = axis
        self.logger = logger

        self._cutframe_idx = None
    
    def get_approximate_cutframe_idxes(self):
        var_threshold = self.params.get("var_threshold", 1)
        skip_num = self.params.get("skip_num", 0)
        corr_thres = self.params.get("corr_thres", None)
        max_var_threshold = self.params.get("max_var_threshold", 200)

        head_appro_idx = find_approximate_single_end(
            self.images_path_list,
            var_threshold=var_threshold,
            max_var_threshold=max_var_threshold,
            corr_thres=corr_thres,
            skip_num=skip_num,
            axis=self.axis)
        tail_appro_idx = find_approximate_single_end(
            self.images_path_list,
            var_threshold=var_threshold,
            max_var_threshold=max_var_threshold,
            corr_thres=corr_thres,
            reverse=True,
            skip_num=skip_num,
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
                               imread=imread_octa,
                               save_path=None):
        if self.cutframe_idx is None:
            raise ValueError("Please provide cutframe index 轴信息.")
        assert len(self.cutframe_idx) == 2

        cover_range = self.params.get("cover_range", 2)
        offset = self.params.get("offset", 0)
        shift = self.params.get("shift", 3)

        # load model
        model = YoloInfer(self.params.model_path,
                          self.device,
                          self.params.imgsz,
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
            
            temp_outputs =[]
            for i in range(shift):
                index = cutframe_idx + i - (shift // 2)
                startline = min(max(0, index - cover_range), len(self.images_path_list) - EPS)
                endline = max(0, min(len(self.images_path_list) - EPS, index + cover_range + 1))

                if startline < endline:
                    img = read_segmented_img(self.images_path_list, startline, endline, imread, axis=self.axis)
                    
                    if save_path:
                        fname = save_path / f"{self.train_sn}_{self.minor_train_code}_{self.channel}_{self.car}_{p}_{i}.jpg"
                        cv2.imwrite(str(fname), img)

                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    temp_outputs = model.infer(img, conf_thres=self.params.conf_thres, iou_thres=self.params.iou_thres)

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
            outputs = [i for i in outputs if self.params.label_translator[int(i[0])] == "end"]
        else:
            outputs = [i for i in outputs if self.params.label_translator[int(i[0])] == "mid"]

        if len(outputs) < 1:
            raise ValueError("Can't find cut line for splitting carriage.")
        
        max_output = select_best_yolobox(outputs, self.params.method)
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

    def _dev_generate_cutpoints_img_(self, save_path, imread=imread_octa, aux=""):
        if self.cutframe_idx is None:
            raise ValueError("Please provide cutframe index 轴信息.")

        cover_range = self.params.get("cover_range", 2)
        offset = self.params.get("offset", 0)
        shift = self.params.get("shift", 3)

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        for p, cutframe_idx in enumerate(self.cutframe_idx):
            if p == 0:
                cutframe_idx -= offset 
            else:
                cutframe_idx += offset 
            for i in range(shift):
                index = cutframe_idx + i - (shift // 2)
                startline = min(max(0, index - cover_range), len(self.images_path_list) - EPS)
                endline = max(0, min(len(self.images_path_list) - EPS, index + cover_range + 1))
                print(index, startline, endline)
                if startline < endline:
                    img = read_segmented_img(self.images_path_list, startline, endline, imread, axis=self.axis)
                    fname = save_path / f"{aux}_{self.minor_train_code}_{self.train_num}_{self.train_sn}_{self.channel}_{self.carriage}_{p}_{i}.jpg"
                    cv2.imwrite(str(fname), img)
                    print(f">>> {fname}.")

    def _dev_generate_car_template_(self, save_path, cutframe_idxes=None, imread=imread_quarter):
        # ---------- generate cut points for training 
        if cutframe_idxes is not None:
            self.update_cutframe_idx(cutframe_idxes[0], cutframe_idxes[1])
        else:
            cutframe_idxes = self.get_approximate_cutframe_idxes()
            # update cutframe idx
            self.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], 
                        cutframe_idxes[self.qtrain_info.carriage])

        save_path = Path(save_path) / self.channel 
        save_path.mkdir(parents=True, exist_ok=True)
        
        test_startline, test_endline = self.get_specific_cutpoints()
        img = read_segmented_img(self.images_path_list, test_startline, test_endline, imread, axis=self.axis)
        fname = save_path / f"car_{self.carriage}.jpg"
        cv2.imwrite(str(fname), img)
        print(f">>> {fname}.")

