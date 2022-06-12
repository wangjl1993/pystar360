
import numpy as np

from omegaconf import OmegaConf
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

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


class Splitter:
    def __init__(self,
                 config,
                 images_path_list,
                 major_train_code,
                 minor_train_code,
                 train_sn,
                 channel,
                 carriage,
                 train_library_path,
                 device,
                 logger=None):
        
        self.config = config
        self.images_path_list = images_path_list
        self.major_train_code = major_train_code
        self.minor_train_code = minor_train_code
        self.train_sn = train_sn
        self.channel = channel
        self.carriage = carriage
        self.device = device
        self.logger = logger
        self.axis = config.axis
        self.train_library_path = Path(train_library_path) / (
            major_train_code + ".yaml")
        self.train_dict = OmegaConf.load(str(
            self.train_library_path)).minor_train_code

        self._cutframe_idx = None
    
    def get_approximate_cutframe_idxes(self, var_threshold=1):
        head_appro_idx = find_approximate_single_end(
            self.images_path_list,
            var_threshold=var_threshold,
            axis=self.axis)
        tail_appro_idx = find_approximate_single_end(
            self.images_path_list,
            var_threshold=var_threshold,
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

    def get_specific_cutpoints(self,
                               offset=1,
                               shift=3,
                               imread=imread_quarter,
                               save_path=None):
        if self.cutframe_idx is None:
            raise ValueError("Please provide cutframe index 轴信息.")
        assert len(self.cutframe_idx) == 2

        # load model
        model = YoloInfer(self.config.model.model_path,
                          self.device,
                          self.config.model.imgsz,
                          logger=self.logger)
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

        _cutpoints = []
        for p, cutframe_idx in enumerate(self.cutframe_idx):
            possible_outputs = []
            for i in range(shift):
                index = cutframe_idx + i - (shift // 2)
                startline = int(max(0, index - offset))
                endline = int(
                    min(len(self.images_path_list), index + offset + 1))
                img = read_segmented_img(self.images_path_list, startline,
                                         endline, imread, axis=self.axis)
                if save_path:
                    fname = save_path / f"{self.train_sn}_{self.minor_train_code}_{self.channel}_{self.car}_{p}_{i}.jpg"
                    cv2.imwrite(str(fname), img)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                temp_outputs = model.infer(conf_thres=self.config.model.conf_thres, iou_thres=self.config.model.iou_thres)
                for out in temp_outputs:
                     # 0:class, 1:ctrx, 2:ctry, 3;w, 4:h, 5:confidence, 6:startline, 7:endline
                    possible_outputs.append(out+[startline, endline])

            new_cutpoints = self._post_process(possible_outputs, p)
            _cutpoints.append(new_cutpoints)
                
        return _cutpoints

    def _post_process(self, outputs, p):
        # return specific cutpoints
        outputs = [i for i in outputs if self.config.model.label_converter[int(i[0])] == "end"]
        if len(outputs) < 1:
            raise ValueError("Can't find cut line for splitting carriage.")
        max_output = max(outputs, key=lambda x: x[5]) # 5: max confidence
        startline = max_output[6] #6: startline
        endline = max_output[7] #7: endline 

        if self.carriage == 1 or self.carriage == self.train_dict.num: 
            # first carraige or last carriage 
            if self.axis == 1:
                # x axis cutline 
                point = yolo_xywh2xyxy(max_output[1:5], startline, 0, 1., endline-startline)
                if p == 0:
                    return point[0][0]
                elif p == 1:
                    return point[1][0]
            elif self.axis == 0:
                # y axis cutline
                point = yolo_xywh2xyxy(max_output[1:5], 0, startline, endline-startline, 1.)
                if p == 0:
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

    def _generate_cutpoints_img(self, save_path, imread=imread_quarter, aux="", offset=1, shift=3):
        if self.cutframe_idx is None:
            raise ValueError("Please provide cutframe index 轴信息.")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        for p, cutframe_idx in enumerate(self.cutframe_idx):
            for i in range(shift):
                index = cutframe_idx + i - (shift // 2)
                startline = int(max(0, index - offset))
                endline = int(
                    min(len(self.images_path_list), index + offset + 1))
                img = read_segmented_img(self.images_path_list, startline,
                                         endline, imread, axis=self.axis)
                fname = save_path / f"{aux}_{self.train_sn}_{self.minor_train_code}_{self.channel}_{self.car}_{p}_{i}.jpg"
                cv2.imwrite(str(fname), img)
