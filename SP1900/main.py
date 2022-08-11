from pathlib import Path
import sys

sys.path.append("..")

import cv2 
from pystar360.base.reader import ImReader
from pystar360.base.splitter import Splitter
from pystar360.base.locator import Locator
from pystar360.base.detector import Detector
from pystar360.base.dataStruct import json2bbox_formater
from pystar360.utilities.fileManger import read_json, read_yaml, write_json
from pystar360.utilities.helper import concat_str, frame2index, imread_tenth, imread_quarter,read_segmented_img,imread_full
from pystar360.utilities.visualizer import plt_bboxes_on_img

import SP1900.local_settings as SETTINGS
from pystar360.robot import pyStar360RobotBase

class pyStar360Robot(pyStar360RobotBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
       
        self.imreader = ImReader(self.qtrain_info.test_train.path, self.qtrain_info.channel, verbose=True, logger=self.logger)
        self.splitter = Splitter(self.qtrain_info, self.channel_params.splitter, self.imreader, 
                        SETTINGS.TRAIN_LIB_PATH, self.device, axis=self.channel_params.axis, logger=self.logger)
        self.locator = Locator(self.qtrain_info, self.channel_params.locator, self.device, axis=self.channel_params.axis, logger=self.logger)
        self.detector = Detector(self.qtrain_info, self.item_params, self.device, axis=self.channel_params.axis, logger=self.logger)

    def run(self):
        # output save path 
        self.output_save_path = SETTINGS.OUTPUT_PATH / concat_str(self.qtrain_info.major_train_code, 
                    self.qtrain_info.minor_train_code, self.qtrain_info.train_num, self.qtrain_info.train_sn)
        self.output_save_path.mkdir(exist_ok=True, parents=True)
        
        # # 如果没有提供有效的轴信息，图片自行寻找
        # # 车厢分割
        # cutframe_idxes = self.splitter.get_approximate_cutframe_idxes()
        # self.splitter.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], 
        #             cutframe_idxes[self.qtrain_info.carriage])
        # test_startline, test_endline = self.splitter.get_specific_cutpoints()

        # # 读图 read full
        # self.qtrain_info.test_train.startline = test_startline
        # self.qtrain_info.test_train.endline = test_endline
        # img_h = self.channel_params.img_h 
        # img_w = self.channel_params.img_w
        # test_img = read_segmented_img(self.imreader, test_startline, test_endline, 
        #         imread_full, axis=self.channel_params.axis)
        
        # #读取模版信息
        # template_path = self.template_path / "template.json"
        # itemInfo = read_json(str(template_path))["carriages"][str(self.qtrain_info.carriage)]

        # self.locator.update_test_traininfo(test_startline, test_endline)
        # self.locator.update_temp_traininfo(itemInfo["startline"], itemInfo["endline"])
       
        # anchor_bboxes = json2bbox_formater(itemInfo.get("anchors", []))
        # anchor_bboxes = self.locator.locate_anchors_yolo(anchor_bboxes, test_img, img_h, img_w)
        # self.locator.get_affine_transformation(anchor_bboxes) # template VS test anchors relationship 

        # extra_anchor_bboxes = json2bbox_formater(itemInfo.get("anchors", [])) # second time location  **** 
        # extra_anchor_bboxes = self.locator.locate_bboxes_according2anchors(extra_anchor_bboxes)
        # extra_anchor_bboxes = self.locator.locate_anchors_yolo(extra_anchor_bboxes, test_img, img_h, img_w,
        #                         use_ratio_adjust=False)
        # self.locator.get_affine_transformation(extra_anchor_bboxes)
        # anchor_bboxes = extra_anchor_bboxes
        # # print(extra_anchor_bboxes) 

        # item_bboxes = json2bbox_formater(itemInfo.get("items", []))
        # item_bboxes = self.locator.locate_bboxes_according2anchors(item_bboxes)
        # chunk_bboxes = json2bbox_formater(itemInfo.get("chunks", []))
        # chunk_bboxes = self.locator.locate_bboxes_according2anchors(chunk_bboxes)
        # template_path = self.template_path / "template.json"
        # itemInfo = read_json(str(template_path))["carriages"][str(self.qtrain_info.carriage)]

        # # 检测 detection
        # item_bboxes = self.detector._dev_item_null_detection_(item_bboxes)

        # json_dict = {}

        # json_dict["train_info"] = self.qtrain_info.to_dict()
        # json_dict["anchors"] = [i.to_dict() for i in anchor_bboxes]
        # json_dict["items"] = [i.to_dict() for i in item_bboxes]
        # json_dict["chunks"] = [i.to_dict() for i in chunk_bboxes]

        # fname = concat_str(self.qtrain_info.channel, self.qtrain_info.carriage)
        # fname = self.output_save_path / (fname + ".json")
        # write_json(fname, json_dict)

        ## print 
        # import cv2 
        # img = plt_bboxes_on_img(item_bboxes, test_img.copy(), img_h, img_w,
        #          test_startline, axis=self.channel_params.axis, vis_lv=1)
        # fname = self.output_save_path/ (concat_str(self.qtrain_info.channel, self.qtrain_info.carriage) + ".jpg")
        # cv2.imwrite(str(fname), img)
        # print(fname)
        # img = plt_bboxes_on_img(anchor_bboxes, test_img.copy(), img_h, img_w,
        #          test_startline, axis=self.channel_params.axis, vis_lv=3)
        # fname = self.output_save_path/ (concat_str(self.qtrain_info.channel, self.qtrain_info.carriage, "anchors") + ".jpg")
        # cv2.imwrite(str(fname), img)
        # print(fname)

        # save_path = Path(SETTINGS.FOLDER_NAME) / "template"
        # self._dev_generate_template_(save_path)

        # save_path = SETTINGS.DATASET_PATH / "new1415cutpoints_test2"  / "images" / "train"
        # self._dev_generate_cutpoints_(save_path)

        save_path = SETTINGS.DATASET_PATH / "1213anchors_extra" / "images" / "train"
        self._dev_generate_anchors_(save_path, label_list=["1"])

       
