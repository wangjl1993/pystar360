

from pathlib import Path
import sys
sys.path.append("..")

from uni360detection.base.reader import ImReader
from uni360detection.base.splitter import Splitter
from uni360detection.utilities.fileManger import read_yaml
from uni360detection.utilities.helper import frame2index, imread_tenth, imread_quarter,read_segmented_img
import cv2 

import CR300AF.local_settings as SETTINGS

class pyStar360Robot:
    def __init__(self, qtrain_info, channel_param, item_params, device="cpu"):

        self.qtrain_info = qtrain_info 
        self.channel_params = read_yaml(channel_param)[str(qtrain_info.channel)]
        self.item_params = read_yaml(item_params)
        # self.itemInfo = itemInfo

        self.imreader = ImReader(qtrain_info.test_train.path, qtrain_info.channel, verbose=True)
        self.splitter = Splitter(qtrain_info, self.channel_params.splitter, self.imreader, 
                        SETTINGS.TRAIN_LIB_PATH, device, axis=self.channel_params.axis)
        # self.locator = Locator(qtrain_info, self.channel_params.locator, itemInfo, device, axis=self.channel_params.axis)

    def run(self):
        cutframe_idxes = self.splitter.get_approximate_cutframe_idxes()
        self.splitter.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], cutframe_idxes[self.qtrain_info.carriage])
        # self.splitter._generate_cutpoints_img("./dataset/cutframe")
        
        # ---------- generate cut points
        cutpoints = self.splitter.get_specific_cutpoints()
        print(cutpoints)
        img = read_segmented_img(self.imreader, cutpoints[0], cutpoints[1], imread_quarter, axis=self.channel_params.axis)
        save_path = Path(f"./dataset/{self.qtrain_info.minor_train_code}/{self.qtrain_info.channel}")
        save_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path / (f"car_{self.qtrain_info.carriage}.jpg")), img)
        # return cutpoints
    
        # ---------- vis cutpoitns
        # for i in cutpoints:
        #     index, shift = frame2index(i, self.channel_params.img_w)
        #     img = imread_tenth(self.imreader[index])
        #     img_h, img_w = img.shape 
        #     newx = int(shift * imread_tenth.resize_ratio)
        #     img = cv2.line(img, (newx, 0), (newx, img_h), (255, 0, 0), 1) 
        #     uid = datetime.now().strftime(DEFAULT_STR_DATETIME) 
        #     cv2.imwrite(f"./dataset/{uid}.jpg", img)