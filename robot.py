import cv2
from pystar360.robot import pyStar360RobotBase
import top.local_settings as SETTINGS
from pystar360.utilities.helper import (
    concat_str, imread_full,
    crop_segmented_rect, frame2rect,
)
from pystar360.utilities.visualizer import plt_bboxes_on_img
from pystar360.utilities.helper3d import imread3d_hx
from pystar360.utilities.fileManger import read_json, read_yaml
from pystar360.base.dataStruct import json2bbox_formater
from pystar360.base.locator import Locator
from pystar360.base.splitter import Splitter
from pystar360.base.reader import BatchImReader
from pystar360.base.detector import Detector
import numpy as np
import os
from pystar360.utilities.logger import w_logger

###  develop example
class pyStar360DevRobot(pyStar360RobotBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_params_fpath = SETTINGS.CHANNEL_PARAMS_PATH / self.qtrain_info.minor_train_code / "channel_params.yaml"
        self.item_params_fpath = SETTINGS.ITEM_PARAMS_PATH / self.qtrain_info.minor_train_code / "item_params.yaml"
        self.channel_params = read_yaml(self.channel_params_fpath, key=SETTINGS.MAC_PASSWORD)[str(self.qtrain_info.channel)]
        self.item_params = read_yaml(self.item_params_fpath, key=SETTINGS.MAC_PASSWORD)
        
        # 根据自己路径读取模板，静态模板
        self.template_dir = SETTINGS.TEMPLATE_PATH / self.qtrain_info.minor_train_code
        template_path = self.template_dir / f"template_{self.qtrain_info.channel}.json"
        self.itemInfo = read_json(template_path, key=SETTINGS.MAC_PASSWORD)["carriages"][str(self.qtrain_info.carriage)]
        static_template_path = self.template_dir / "chunk_items_template.json"
        self.static_template = read_json(static_template_path, key=SETTINGS.MAC_PASSWORD) if static_template_path.exists() else None

        self.imreader = BatchImReader(
            qtrain_info=self.qtrain_info,
            client='HUAXING',
            debug=True
        )
        self.splitter = Splitter(
            self.qtrain_info, self.channel_params.splitter, self.imreader, SETTINGS.TRAIN_LIB_PATH, 
            self.device, axis=self.channel_params.axis, debug=True, mac_password=SETTINGS.MAC_PASSWORD
        )
        self.locator = Locator(
            qtrain_info=self.qtrain_info, local_params=self.channel_params.locator, device=self.device, 
            axis=self.channel_params.axis, debug=True, mac_password=SETTINGS.MAC_PASSWORD
        )
        self._init_carriage_info()

    def _init_carriage_info(self, ):
        if self.qtrain_info.curr_train2d is not None:
            self.qtrain_info.curr_train2d.img_h = self.channel_params.img_h
            self.qtrain_info.curr_train2d.img_w = self.channel_params.img_w

        if self.qtrain_info.curr_train3d is not None:
            self.qtrain_info.curr_train3d.img_h = self.channel_params.img3d_h
            self.qtrain_info.curr_train3d.img_w = self.channel_params.img3d_w

        if self.qtrain_info.hist_train2d is not None:
            self.qtrain_info.hist_train2d.img_h = self.channel_params.img_h
            self.qtrain_info.hist_train2d.img_w = self.channel_params.img_w
        
        if self.qtrain_info.hist_train3d is not None:
            self.qtrain_info.hist_train3d.img_h = self.channel_params.img3d_h
            self.qtrain_info.hist_train3d.img_w = self.channel_params.img3d_w

    def run(self):

        cfg = read_yaml("develop_cfg.yaml")
        crop_seg = cfg.get("crop_seg", False)
        generate_template = cfg.get("generate_template", False)
        crop_anchors = cfg.get("crop_anchors", False) 
        crop_items = cfg.get("crop_items", [])

        #  crop head/tail/connector 
        if crop_seg:
            cutframe_idxes = self.splitter.get_approximate_cutframe_idxes()
            self.splitter.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], cutframe_idxes[self.qtrain_info.carriage])
            output_save_path = SETTINGS.DATASET_PATH / concat_str(self.qtrain_info.major_train_code, f"{self.qtrain_info.channel}" ,"seg_dataset")
            output_save_path.mkdir(exist_ok=True, parents=True)
            self.splitter._dev_generate_cutpoints_img_(output_save_path)

        if generate_template:
            cutframe_idxes = self.splitter.get_approximate_cutframe_idxes()
            self.splitter.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], cutframe_idxes[self.qtrain_info.carriage])
            output_save_path = SETTINGS.OUTPUT_PATH / concat_str(self.qtrain_info.major_train_code, "template", self.qtrain_info.train_sn)
            self.splitter._dev_generate_car_template_(output_save_path)
    
        if crop_anchors:
            output_save_path = SETTINGS.DATASET_PATH / concat_str(self.qtrain_info.major_train_code, f"{self.qtrain_info.channel}" ,"anchors_dataset")
            output_save_path.mkdir(exist_ok=True, parents=True)
            self.splitter.run(imread_full)
            self.locator.update_temp_traininfo(self.itemInfo["startline"], self.itemInfo["endline"])                        
            self.locator.update_test_traininfo(self.qtrain_info.curr_train2d.startline, self.qtrain_info.curr_train2d.endline)
            anchors_bbox = json2bbox_formater(self.itemInfo.get("anchors", []))
            self.locator._dev_generate_anchors_img_(
                anchors_bbox, output_save_path, self.qtrain_info.curr_train2d.img, 
                self.qtrain_info.curr_train2d.img_h, self.qtrain_info.curr_train2d.img_w
            )


        if len(crop_items) > 0:
            output_save_path = SETTINGS.DATASET_PATH / concat_str(self.qtrain_info.major_train_code, "items_dataset")
            output_save_path.mkdir(exist_ok=True, parents=True)

            if self.qtrain_info.curr_train2d.img is None:
                self.splitter.run(imread_full)
            
            anchor_bboxes = json2bbox_formater(self.itemInfo.get("anchors", []))
            item_bboxes = json2bbox_formater(self.itemInfo.get("items", []))
            static_chunk_bboxes = json2bbox_formater(self.itemInfo.get("static_chunks", []))
            dynamic_chunk_bboxes = json2bbox_formater(self.itemInfo.get("dynamic_chunks", []))
            self.locator.update_temp_traininfo(self.itemInfo["startline"], self.itemInfo["endline"])
            self.locator.run(anchor_bboxes, item_bboxes, static_chunk_bboxes, dynamic_chunk_bboxes, self.static_template)

            all_bboxes = self.qtrain_info.item_bboxes
            for bbox in all_bboxes:
                name = bbox.name
                if name in crop_items:
                    proposal_rect_f = bbox.curr_proposal_rect
                    proposal_rect_p = frame2rect(
                        proposal_rect_f, self.qtrain_info.curr_train2d.startline, self.qtrain_info.curr_train2d.img_h, 
                        self.qtrain_info.curr_train2d.img_w, axis=self.channel_params.axis
                    )
                    img = crop_segmented_rect(self.qtrain_info.curr_train2d.img, proposal_rect_p)
                    save_root = output_save_path / name
                    save_root.mkdir(exist_ok=True, parents=True)
                    n = len(os.listdir(save_root))
                    save_name = save_root / f"{self.qtrain_info.train_num}-{self.qtrain_info.train_sn}-{self.qtrain_info.channel}-{self.qtrain_info.carriage}-{n:04}.jpg"
                    cv2.imwrite(str(save_name), img)


### deploy example
class pyStar360DepRobot(pyStar360RobotBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_params_fpath = SETTINGS.CHANNEL_PARAMS_PATH / self.qtrain_info.minor_train_code / "channel_params.yaml"
        self.item_params_fpath = SETTINGS.ITEM_PARAMS_PATH / self.qtrain_info.minor_train_code / "item_params.yaml"
        self.channel_params = read_yaml(self.channel_params_fpath, key=SETTINGS.MAC_PASSWORD)[str(self.qtrain_info.channel)]
        self.item_params = read_yaml(self.item_params_fpath, key=SETTINGS.MAC_PASSWORD)
        
        # 根据自己路径读取模板，静态模板
        self.template_dir = SETTINGS.TEMPLATE_PATH / self.qtrain_info.minor_train_code
        template_path = self.template_dir / f"template_{self.qtrain_info.channel}.json"
        self.itemInfo = read_json(template_path, key=SETTINGS.MAC_PASSWORD)["carriages"][str(self.qtrain_info.carriage)]
        static_template_path = self.template_dir / "chunk_items_template.json"
        self.static_template = read_json(static_template_path, key=SETTINGS.MAC_PASSWORD) if static_template_path.exists() else None

        try:
            self.imreader = BatchImReader(
                qtrain_info=self.qtrain_info,
                client='HUAXING'
            )
            self.splitter = Splitter(
                self.qtrain_info, self.channel_params.splitter, self.imreader, SETTINGS.TRAIN_LIB_PATH, 
                self.device, axis=self.channel_params.axis, mac_password=SETTINGS.MAC_PASSWORD
            )

            self.locator = Locator(
                qtrain_info=self.qtrain_info, local_params=self.channel_params.locator, device=self.device, 
                axis=self.channel_params.axis, mac_password=SETTINGS.MAC_PASSWORD
            )
            self.detector = Detector(
                qtrain_info=self.qtrain_info, item_params=self.item_params, device=self.device, 
                axis=self.channel_params.axis, mac_password=SETTINGS.MAC_PASSWORD
            )
            self._init_carriage_info()
        except Exception as e:
            w_logger.error(e)
            w_logger.exception(e)

    def _init_carriage_info(self, ):
        if self.qtrain_info.curr_train2d is not None:
            self.qtrain_info.curr_train2d.img_h = self.channel_params.img_h
            self.qtrain_info.curr_train2d.img_w = self.channel_params.img_w

        if self.qtrain_info.curr_train3d is not None:
            self.qtrain_info.curr_train3d.img_h = self.channel_params.img3d_h
            self.qtrain_info.curr_train3d.img_w = self.channel_params.img3d_w

        if self.qtrain_info.hist_train2d is not None:
            self.qtrain_info.hist_train2d.img_h = self.channel_params.img_h
            self.qtrain_info.hist_train2d.img_w = self.channel_params.img_w
        
        if self.qtrain_info.hist_train3d is not None:
            self.qtrain_info.hist_train3d.img_h = self.channel_params.img3d_h
            self.qtrain_info.hist_train3d.img_w = self.channel_params.img3d_w
        

    def run(self):
        self.output_save_path = SETTINGS.OUTPUT_PATH / concat_str(
            self.qtrain_info.major_train_code,
            self.qtrain_info.minor_train_code, 
            self.qtrain_info.train_num, 
            self.qtrain_info.train_sn
        )
        self.output_save_path.mkdir(exist_ok=True, parents=True)
        try:
            self.splitter.run(imread_full, imread3d_hx)

            self.locator.update_temp_traininfo(self.itemInfo["startline"], self.itemInfo["endline"])
            anchor_bboxes = json2bbox_formater(self.itemInfo.get("anchors", []))
            item_bboxes = json2bbox_formater(self.itemInfo.get("items", []))
            static_chunk_bboxes = json2bbox_formater(self.itemInfo.get("static_chunks", []))
            dynamic_chunk_bboxes = json2bbox_formater(self.itemInfo.get("dynamic_chunks", []))
            self.locator.run(anchor_bboxes, item_bboxes, static_chunk_bboxes, dynamic_chunk_bboxes, self.static_template)

            
            map3d_minor_axis_affine_matrix = self.channel_params.locator.get('map3d_minor_axis_affine_matrix', [1, 0])
            map3d_major_axis_affine_matrix = self.channel_params.locator.get('map3d_major_axis_affine_matrix', [1, 0])
            map3d_minor_axis_poly_func = np.poly1d(map3d_minor_axis_affine_matrix)
            map3d_major_axis_poly_func = np.poly1d(map3d_major_axis_affine_matrix)
            res = self.detector.detect_items(
                map3d_minor_axis_poly_func=map3d_minor_axis_poly_func, 
                map3d_major_axis_poly_func=map3d_major_axis_poly_func
            )
        except Exception as e:
            w_logger.error(e)
            w_logger.exception(e)

        img = plt_bboxes_on_img(
            res, 
            self.qtrain_info.curr_train2d.img.copy(), 
            self.qtrain_info.curr_train2d.img_h, 
            self.qtrain_info.curr_train2d.img_w,
            self.qtrain_info.curr_train2d.startline, 
            axis=self.channel_params.axis, 
            vis_lv=1,
            rect='curr_rect'
        )
        fname = self.output_save_path / (concat_str(self.qtrain_info.channel, self.qtrain_info.carriage) + "_curr2d.jpg")
        cv2.imwrite(str(fname), img)
        print(fname)
    
        # img = plt_bboxes_on_img(
        #     res, 
        #     self.qtrain_info.hist_train2d.img.copy(), 
        #     self.qtrain_info.hist_train2d.img_h, 
        #     self.qtrain_info.hist_train2d.img_w,
        #     self.qtrain_info.hist_train2d.startline, 
        #     axis=self.channel_params.axis, 
        #     vis_lv=1,
        #     rect='hist_rect'
        # )
        # fname = self.output_save_path / (concat_str(self.qtrain_info.channel, self.qtrain_info.carriage) + "_hist2d.jpg")
        # cv2.imwrite(str(fname), img)
        # print(fname)


        # img = plt_bboxes_on_img(
        #     res, 
        #     self.qtrain_info.curr_train3d.img.copy(), 
        #     self.qtrain_info.curr_train3d.img_h, 
        #     self.qtrain_info.curr_train3d.img_w,
        #     self.qtrain_info.curr_train3d.startline, 
        #     axis=self.channel_params.axis, 
        #     vis_lv=1,
        #     rect='curr_rect3d',
        #     resize_ratio=0.5
        # )
        # fname = self.output_save_path / (concat_str(self.qtrain_info.channel, self.qtrain_info.carriage) + "_curr3d.jpg")
        # cv2.imwrite(str(fname), img)
        # print(fname)
    
        # img = plt_bboxes_on_img(
        #     res, 
        #     self.qtrain_info.hist_train3d.img.copy(), 
        #     self.qtrain_info.hist_train3d.img_h, 
        #     self.qtrain_info.hist_train3d.img_w,
        #     self.qtrain_info.hist_train3d.startline, 
        #     axis=self.channel_params.axis, 
        #     vis_lv=1,
        #     rect='hist_rect3d',
        #     resize_ratio=0.5
        # )
        # fname = self.output_save_path / (concat_str(self.qtrain_info.channel, self.qtrain_info.carriage) + "_hist3d.jpg")
        # cv2.imwrite(str(fname), img)
        # print(fname)
