from pathlib import Path
import sys

sys.path.append("..")

import cv2 
from pystar360.base.reader import ImReader
from pystar360.base.splitter import Splitter
from pystar360.base.locator import Locator
from pystar360.base.detector import Detector
from pystar360.base.dataStruct import json2bbox_formater
from pystar360.utilities.fileManger import read_json, read_yaml
from pystar360.utilities.helper import concat_str, frame2index, imread_tenth, imread_quarter,read_segmented_img,imread_full
from pystar360.utilities.visualizer import plt_bboxes_on_img

import SP1900.local_settings as SETTINGS
from pystar360.robot import pyStar360RobotBase

class pyStar360Robot(pyStar360RobotBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # output save path 
        self.output_save_path = SETTINGS.OUTPUT_PATH / concat_str(self.qtrain_info.major_train_code, 
                    self.qtrain_info.minor_train_code, self.qtrain_info.train_num, self.qtrain_info.train_sn)
        self.output_save_path.mkdir(exist_ok=True, parents=True)

        # template_path = self.template_path / "template.json"
        # self.itemInfo = read_json(str(template_path))["carriages"][str(self.qtrain_info.carriage)]
        self.imreader = ImReader(self.qtrain_info.test_train.path, self.qtrain_info.channel, verbose=True, logger=self.logger)
        self.splitter = Splitter(self.qtrain_info, self.channel_params.splitter, self.imreader, 
                        SETTINGS.TRAIN_LIB_PATH, self.device, axis=self.channel_params.axis, logger=self.logger)
        # self.locator = Locator(self.qtrain_info, self.channel_params.locator, self.itemInfo, 
        #                         self.device, axis=self.channel_params.axis, logger=self.logger)
        # self.detector = Detector(self.qtrain_info, self.item_params, self.itemInfo, self.device,
        #                         axis=self.channel_params.axis, logger=self.logger)

    def run(self):
        # save_path = SETTINGS.LOCAL_OUTPUT_PATH / concat_str(self.qtrain_info.major_train_code, 
        #             self.qtrain_info.minor_train_code, self.qtrain_info.train_num, self.qtrain_info.train_sn)
        # save_path.mkdir(exist_ok=True, parents=True)

        # # 如果没有提供有效的轴信息，图片自行寻找
        # # 车厢分割
        # cutframe_idxes = self.splitter.get_approximate_cutframe_idxes()
        # self.splitter.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], 
        #             cutframe_idxes[self.qtrain_info.carriage])
        # test_startline, test_endline = self.splitter.get_specific_cutpoints()

        # # 读图 full
        # self.qtrain_info.test_train.startline = test_startline
        # self.qtrain_info.test_train.endline = test_endline
        # img_h = self.channel_params.img_h 
        # img_w = self.channel_params.img_w
        # test_img = read_segmented_img(self.imreader, test_startline, test_endline, 
        #         imread_full, axis=self.channel_params.axis)
        
        # # 定位
        # self.locator.update_test_traininfo(test_startline, test_endline)
        # self.locator.locate_anchors_yolo(test_img, img_h, img_w)
        # item_bboxes = self.locator.locate_bboxes_according2anchors(bbox_formater(self.itemInfo["items"]))

        # # 检测
        # item_bboxes = self.detector.detect_items(item_bboxes, test_img, test_startline, img_w, img_h)

        # # print 
        # img = plt_bboxes_on_img(item_bboxes, test_img, img_h, img_w,
        #          test_startline, axis=self.channel_params.axis, vis_lv=1)
        # fname = save_path/ (concat_str(self.qtrain_info.channel, self.qtrain_info.carriage) + ".jpg")
        # cv2.imwrite(str(fname), img)
        # save_path = f"./{SETTINGS.FOLDER_NAME}/template"
        # self._dev_generate_template_(save_path)

        save_path = SETTINGS.DATASET_PATH / "120cutpoints"
        self._dev_generate_cutpoints_(save_path)