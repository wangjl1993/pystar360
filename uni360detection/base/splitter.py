import numpy as np

from omegaconf import OmegaConf
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

from uni360detection.utilities.helper import *


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
                 images_path_list,
                 major_train_code,
                 minor_train_code,
                 train_sn,
                 channel,
                 carriage,
                 train_library_path,
                 axis=1,
                 var_threshold=1):
        self.images_path_list = images_path_list
        self.major_train_code = major_train_code
        self.minor_train_code = minor_train_code
        self.train_sn = train_sn
        self.channel = channel
        self.carriage = carriage
        self.axis = axis
        self.var_threshold = var_threshold
        self.train_library_path = Path(train_library_path) / (
            major_train_code + ".yaml")
        self.train_dict = OmegaConf.load(str(
            self.train_library_path)).minor_train_code

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

    def get_specific_cutpoints(self, offset=1, shift=3, save_path=None):
        if self.cutframe_idx is None:
            raise ValueError("Please provide cutframe index 轴信息.")

        # load model

        self._cutpoints = []
        for p in range(len(self.cutframe_idx)):
            possible_output = []
            for i in range(shift):
                index = self.cutframe_idx[p] + i - (shift // 2)
                start_l = int(max(0, index - offset))
                start_r = int(
                    min(len(self.images_path_list), index + offset + 1))
                image = read_segmented_img(self.images_path_list, start_l,
                                           start_r, imread_full)

    def _post_process(self):
        pass

    def _generate_cutpoints_img(self, save_path, aux="", offset=1, shift=3):
        if self.cutframe_idx is None:
            raise ValueError("Please provide cutframe index 轴信息.")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        for p in range(len(self.cutframe_idx)):
            for i in range(shift):
                index = self.cutframe_idx[p] + i - (shift // 2)
                start_l = int(max(0, index - offset))
                start_r = int(
                    min(len(self.images_path_list), index + offset + 1))
                image = read_segmented_img(self.images_path_list, start_l,
                                           start_r, imread_full)
                fname = save_path / f"{self.train_sn}_{self.minor_train_code}_{self.channel}_{self.car}_{p}_{i}.jpg"
                cv2.imwrite(str(fname), image)
