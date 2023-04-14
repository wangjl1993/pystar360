import functools
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

from pystar360.base.dataStruct import QTrainInfo
from pystar360.yolo.inference import YoloInfer, yolo_xywh2xyxy, select_best_yolobox
from pystar360.utilities.fileManger import read_yaml
from pystar360.utilities.helper import *
from pystar360.utilities._logger import d_logger
from pystar360.base.reader import BatchImReader
from copy import deepcopy
from typing import Callable, Optional

EPS = 1e-6  # eplison
MAXSIZE_CACHE = 8  # cache number
DEFAULT_VAR_THRESHOLD = 1
DEFAULT_SKIP_NUM = 0
DEFAULT_CORR_THRES = None
DEFAULT_MAX_VAR_THRESHOLD = 5000
DEFAULT_COVER_RANGE = 2  # 覆盖范围
DEFAULT_OFFSET = 0  # 偏移量
DEFAULT_SHIFT = 3  # 移动次数
DEFAULT_STEP = 1  # 移动步长


__all__ = ["find_approximate_single_end", "Splitter"]


@functools.lru_cache(maxsize=MAXSIZE_CACHE)
def find_approximate_single_end(
    l,
    var_threshold=1,
    max_var_threshold=5000,
    corr_thres=None,
    reverse=False,
    axis=1,
    skip_num=0,
    imread=imread_tenth,
    debug=False,
):
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
        raise ValueError(">>> 图像数据质量可能存在以下问题，1、光照过度曝光或过暗；2、拍摄是否完整；")

    if debug:
        d_logger.info(f">>> Aprroximate ends frame: {str(l[idx])}; variation {var1}; direction: {reverse}")
    return idx


class Splitter:
    def __init__(
        self,
        qtrain_info: QTrainInfo,
        local_params,
        batchImreader: BatchImReader,
        train_library_path,
        device,
        axis=1,
        logger=None,
        debug=False,
        mac_password=None,
    ):

        # query information
        self.qtrain_info = qtrain_info

        # local params
        self.params = local_params

        # images list
        self.batchImreader = batchImreader

        train_library_path = Path(train_library_path) / (qtrain_info.major_train_code + ".yaml")
        self.train_dict = read_yaml(str(train_library_path))[self.qtrain_info.minor_train_code]

        self.device = device
        self.axis = axis
        self.logger = logger
        self.debug = debug
        self.mac_password = mac_password

        self._cutframe_idx = None

    def get_approximate_cutframe_idxes(self, images_path_list=None):

        if images_path_list is None:
            images_path_list = self.batchImreader.test_img2d_list 

        var_threshold = self.params.get("var_threshold", DEFAULT_VAR_THRESHOLD)
        skip_num = self.params.get("skip_num", DEFAULT_SKIP_NUM)
        corr_thres = self.params.get("corr_thres", DEFAULT_CORR_THRES)
        max_var_threshold = self.params.get("max_var_threshold", DEFAULT_MAX_VAR_THRESHOLD)

        head_appro_idx = find_approximate_single_end(
            images_path_list,
            var_threshold=var_threshold,
            max_var_threshold=max_var_threshold,
            corr_thres=corr_thres,
            skip_num=skip_num,
            axis=self.axis,
            debug=self.debug,
        )
        tail_appro_idx = find_approximate_single_end(
            images_path_list,
            var_threshold=var_threshold,
            max_var_threshold=max_var_threshold,
            corr_thres=corr_thres,
            reverse=True,
            skip_num=skip_num,
            axis=self.axis,
            debug=self.debug,
        )
        n_frames = tail_appro_idx - head_appro_idx + 1
        default_cutpoints = np.array(self.train_dict.cutpoints)
        cutframe_idxes = np.rint(n_frames * default_cutpoints + head_appro_idx)

        # e.g.
        # cutframe_idxes[self.car-1], cutframe_idxes[self.car]
        return cutframe_idxes

    @property
    def cutframe_idx(self):
        return self._cutframe_idx

    @cutframe_idx.setter
    def cutframe_idx(self, cutframe_idx):
        self._cutframe_idx = cutframe_idx

    def update_cutframe_idx(self, *cutframe_idx):
        if len(cutframe_idx) != 2:
            raise ValueError()

        if self.debug:
            d_logger.info(f">>> Given cutframe index: {cutframe_idx}")
        self.cutframe_idx = cutframe_idx

    def get_specific_cutpoints(self, images_path_list=None, imread=imread_octa, save_path=None):
        if self.cutframe_idx is None:
            raise ValueError("Please provide cutframe index 轴信息.")
        assert len(self.cutframe_idx) == 2

        if images_path_list is None:
            images_path_list = self.batchImreader.test_img2d_list 

        cover_range = self.params.get("cover_range", DEFAULT_COVER_RANGE)
        offset = self.params.get("offset", DEFAULT_OFFSET)
        shift = self.params.get("shift", DEFAULT_SHIFT)
        step = self.params.get("step", DEFAULT_STEP)

        # load model
        model = YoloInfer(
            self.params.model_path, self.device, self.params.imgsz, logger=self.logger, mac_password=self.mac_password
        )

        # save path if needed
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

            temp_outputs = []
            for i in range(shift):
                index = cutframe_idx - ((shift // 2) * step) + (i * step)
                # startline = min(max(0, index - cover_range), len(images_path_list) - EPS)
                # endline = max(0, min(len(images_path_list) - EPS, index + cover_range + 1))
                startline = index - cover_range
                endline = index + cover_range + 1
                if startline < 0:
                    startline = 0
                    endline = startline + (2 * cover_range + 1)
                if endline > len(images_path_list) - 1:
                    endline = len(images_path_list) - EPS
                    startline = endline - (2 * cover_range + 1)

                if self.debug:  # DEBUG
                    d_logger.info(f">>> Order: {p}; Start cutline: {startline}, End cutline: {endline};")

                if startline < endline:
                    img = read_segmented_img(images_path_list, startline, endline, imread, axis=self.axis)

                    if save_path:
                        fname = (
                            save_path
                            / f"{self.qtrain_info.train_sn}_{self.qtrain_info.minor_train_code}_{self.qtrain_info.channel}_{self.qtrain_info.car}_{p}_{i}.jpg"
                        )
                        cv2.imwrite(str(fname), img)

                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    temp_outputs = model.infer(img, conf_thres=self.params.conf_thres, iou_thres=self.params.iou_thres)

                for out in temp_outputs:
                    # 0:class, 1:ctrx, 2:ctry, 3;w, 4:h, 5:confidence, 6:startline, 7:endline
                    out = list(out)
                    possible_outputs.append(out + [startline, endline])

            new_cutpoints = self._post_process(possible_outputs, p)
            _cutpoints.append(new_cutpoints)

        return _cutpoints

    def _post_process(self, outputs, p):
        # return specific cutpoints
        if (self.qtrain_info.carriage == 1 and p == 0) or (self.qtrain_info.carriage == self.train_dict.num and p == 1):
            outputs = [i for i in outputs if self.params.label_translator[int(i[0])] == "end"]
        else:
            outputs = [i for i in outputs if self.params.label_translator[int(i[0])] == "mid"]

        if len(outputs) < 1:
            raise ValueError(">>> 没有精确分割单节车厢，图像可能存在过度畸变问题")

        max_output = select_best_yolobox(outputs, self.params.method)
        startline = max_output[6]  # 6: startline
        endline = max_output[7]  # 7: endline
        if self.qtrain_info.carriage == 1 or self.qtrain_info.carriage == self.train_dict.num:
            # first carraige or last carriage
            if self.axis == 1:
                # x axis cutline
                point = yolo_xywh2xyxy(max_output[1:5], startline, 0, 1.0, endline - startline)
                if p == 0:
                    if self.qtrain_info.carriage == self.train_dict.num:
                        return point[1][0]
                    else:
                        return point[0][0]
                elif p == 1:
                    return point[1][0]
            elif self.axis == 0:
                # y axis cutline
                point = yolo_xywh2xyxy(max_output[1:5], 0, startline, endline - startline, 1.0)
                if p == 0:
                    if self.qtrain_info.carriage == self.train_dict.num:
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
                point = yolo_xywh2xyxy(max_output[1:5], startline, 0, 1.0, endline - startline)
                return point[1][0]
            elif self.axis == 0:
                point = yolo_xywh2xyxy(max_output[1:5], 0, startline, endline - startline, 1.0)
                return point[1][1]
            else:
                raise ValueError(f"Axis {self.axis} is not available.")

    def _dev_generate_cutpoints_img_(self, save_path, images_path_list=None, imread=imread_tenth, aux=""):
        # generate cutpoints image for development
        if self.cutframe_idx is None:
            raise ValueError("Please provide cutframe index 轴信息.")

        if images_path_list is None:
            images_path_list = self.batchImreader.test_img2d_list

        cover_range = self.params.get("cover_range", DEFAULT_COVER_RANGE)
        offset = self.params.get("offset", DEFAULT_OFFSET)
        shift = self.params.get("shift", DEFAULT_SHIFT)
        step = self.params.get("step", DEFAULT_STEP)

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.qtrain_info.carriage == 1:
            cutframe_idxes = self.cutframe_idx
        else:
            cutframe_idxes = [self.cutframe_idx[-1]]
        for p, cutframe_idx in enumerate(cutframe_idxes):
            if p == 0:
                cutframe_idx -= offset
            else:
                cutframe_idx += offset
            for i in range(shift):
                index = cutframe_idx - ((shift // 2) * step) + (i * step)
                startline = index - cover_range
                endline = index + cover_range + 1
                if startline < 0:
                    startline = 0
                    endline = startline + (2 * cover_range + 1)
                if endline > len(images_path_list) - 1:
                    endline = len(images_path_list) - EPS
                    startline = endline - (2 * cover_range + 1)

                if startline < endline:
                    img = read_segmented_img(images_path_list, startline, endline, imread, axis=self.axis)
                    fname = (
                        save_path
                        / f"{aux}_{self.qtrain_info.minor_train_code}_{self.qtrain_info.train_num}_{self.qtrain_info.train_sn}_{self.qtrain_info.channel}_{self.qtrain_info.carriage}_{p}_{i}.jpg"
                    )
                    cv2.imwrite(str(fname), img)
                    d_logger.info(f">>> {fname}.")

    def _dev_generate_car_template_(self, save_path, images_path_list=None, cutframe_idxes=None, imread=imread_quarter):

        if images_path_list is None:
            images_path_list = self.batchImreader.test_img2d_list

        # generate cut points for development
        if cutframe_idxes is not None:
            self.update_cutframe_idx(cutframe_idxes[0], cutframe_idxes[1])
        else:
            cutframe_idxes = self.get_approximate_cutframe_idxes()
            # update cutframe idx
            self.update_cutframe_idx(
                cutframe_idxes[self.qtrain_info.carriage - 1], cutframe_idxes[self.qtrain_info.carriage]
            )

        save_path = Path(save_path) / self.qtrain_info.channel
        save_path.mkdir(parents=True, exist_ok=True)

        test_startline, test_endline = self.get_specific_cutpoints()
        img = read_segmented_img(images_path_list, test_startline, test_endline, imread, axis=self.axis)
        fname = save_path / f"car_{self.qtrain_info.carriage}.jpg"
        cv2.imwrite(str(fname), img)
        d_logger.info(f">>> {fname}.")


    def get_approximate_cutframe_idxes_by_yolo(self, images_path_list, imread=imread_tenth):
        """locate head/tail/connector approximate frame by yolo."""

        def locate_ht(reverse):
            if not reverse:
                start_idx = 0
                end_idx = n-infer_frames_len
                step = 1
            else:
                start_idx = N-infer_frames_len-1
                end_idx = N-n 
                step = -1
            res = None 
            for i in range(start_idx, end_idx, step):
                sidx = i
                eidx = i + infer_frames_len
                img = np.concatenate(images_list[sidx:eidx], axis=self.axis)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # 0:class, 1:ctrx, 2:ctry, 3;w, 4:h, 5:confidence
                outputs = model.infer(img, conf_thres=self.params.conf_thres, iou_thres=self.params.iou_thres, classes=0) # classes=0 表车头车尾
                if len(outputs) > 0:
                    _, ctrx, ctry, _, _, _ = outputs[0] # 取1个结果即可
                    if self.axis == 1:
                        res = sidx + infer_frames_len*ctrx
                    else:
                        res = sidx + infer_frames_len*ctry
                    break
            return round(res)

        DEFAULT_TOP_PER = 0.15
        top_per = self.params.get("top_per", DEFAULT_TOP_PER) # 前top_per% 的图
        cover_range = self.params.get("cover_range")
        infer_frames_len = 2*cover_range + 1
        
        images_list = deepcopy(images_path_list)
        N = len(images_list)
        n = int(N*top_per)

        for i in range(n):
            images_list[i] = imread(images_list[i])
            images_list[N-i-1] = imread(images_list[N-i-1])
        
        model = YoloInfer(
            self.params.model_path, self.device, self.params.imgsz, logger=self.logger, mac_password=self.mac_password
        )

        head_appro_idx = locate_ht(False)
        tail_appro_idx = locate_ht(True)

        if head_appro_idx is None:
            raise ValueError(">>> 未能找到车头大致帧，请检查数据")
        if tail_appro_idx is None:
            raise ValueError(">>> 未能找到车尾大致帧，请检查数据")
        
        n_frames = tail_appro_idx - head_appro_idx + 1
        default_cutpoints = np.array(self.train_dict.cutpoints)
        cutframe_idxes = np.rint(n_frames * default_cutpoints + head_appro_idx)
        return cutframe_idxes


    def _run2d(self, imread2d):
        if self.batchImreader.test_img2d_list is not None:
            cutframe_idxes = self.get_approximate_cutframe_idxes_by_yolo(self.batchImreader.test_img2d_list, imread2d) # self.get_approximate_cutframe_idxes()
            self.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], cutframe_idxes[self.qtrain_info.carriage])
            startline, endline = self.get_specific_cutpoints(self.batchImreader.test_img2d_list, imread2d)
            img = read_segmented_img(self.batchImreader.test_img2d_list, startline, endline, imread2d, axis=self.axis)
            self.qtrain_info.curr_train2d.startline = startline
            self.qtrain_info.curr_train2d.endline = endline 
            self.qtrain_info.curr_train2d.img = img
        
        if self.batchImreader.hist_img2d_list is not None:
            cutframe_idxes = self.get_approximate_cutframe_idxes_by_yolo(self.batchImreader.hist_img2d_list, imread2d) # self.get_approximate_cutframe_idxes()
            self.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], cutframe_idxes[self.qtrain_info.carriage])
            startline, endline = self.get_specific_cutpoints(self.batchImreader.hist_img2d_list, imread2d)
            img = read_segmented_img(self.batchImreader.hist_img2d_list, startline, endline, imread2d, axis=self.axis)
            self.qtrain_info.hist_train2d.startline = startline
            self.qtrain_info.hist_train2d.endline = endline 
            self.qtrain_info.hist_train2d.img = img

    def _run3d(self, imread3d, offset=0):
        """read startline/endline from carriage2D and split carriage3D """
        
        if self.batchImreader.test_img3d_list is not None:
            assert self.qtrain_info.curr_train2d is not None, "you should split Carriage2D first."
            startline = self.qtrain_info.curr_train2d.startline + offset
            endline = self.qtrain_info.curr_train2d.endline + offset 
            img = read_segmented_img(self.batchImreader.test_img3d_list, startline, endline, imread3d, axis=self.axis)
            self.qtrain_info.curr_train3d.startline = startline
            self.qtrain_info.curr_train3d.endline = endline
            self.qtrain_info.curr_train3d.img = img
        
        if self.batchImreader.hist_img3d_list is not None:
            assert self.qtrain_info.hist_train2d is not None, "you should split Carriage2D first."
            startline = self.qtrain_info.hist_train2d.startline + offset
            endline = self.qtrain_info.hist_train2d.endline + offset
            img = read_segmented_img(self.batchImreader.hist_img3d_list, startline, endline, imread3d, axis=self.axis)
            self.qtrain_info.hist_train3d.startline = startline
            self.qtrain_info.hist_train3d.endline = endline
            self.qtrain_info.hist_train3d.img = img
        

    def run(self, imread2d: Callable, imread3d: Optional[Callable]=None):
        """split test2d/test3d/hist2d/hist3d carriage and save information to qtrain_info."""

        self._run2d(imread2d)
        self._run3d(imread3d)