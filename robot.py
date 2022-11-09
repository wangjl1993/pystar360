
import cv2
from pathlib import Path
from pystar360.base.dataStruct import json2bbox_formater
from pystar360.utilities.fileManger import write_json, read_json
from pystar360.utilities.deviceController import get_torch_device, get_environ_info
from pystar360.utilities.helper import (concat_str, read_segmented_img, imread_full, crop_segmented_rect,
    imread_tenth, imread_quarter, frame2rect)
from pystar360.utilities.misc import TryExcept
from pystar360.utilities._logger import d_logger

import pystar360.global_settings as SETTINGS

################################################################################
#### 这个robot base只是用于开发例子展示，不一定非要使用这个类
################################################################################
__all__ = ["pyStar360RobotBase"]
class pyStar360RobotBase:
    def __init__(self, qtrain_info, device="cpu", logger=None):
        self.qtrain_info = qtrain_info 
        self.device = get_torch_device(device) 
        self.logger = logger 
        if self.logger:
            self.logger.info(f">>> Environment device: {get_environ_info()}")
            self.logger.info(f">>> Using device: {self.device}")
        else:
            d_logger.info(f">>> Environment device: {get_environ_info()}")
            d_logger.info(f">>> Using device: {self.device}")

    ###  example
    #     # output save path 
    #     self.output_save_path = SETTINGS.LOCAL_OUTPUT_PATH / concat_str(self.qtrain_info.major_train_code, 
    #                 self.qtrain_info.minor_train_code, self.qtrain_info.train_num, self.qtrain_info.train_sn)
    #     self.output_save_path.mkdir(exist_ok=True, parents=True)

    #     template_path = self.template_path / "template.json"
    #     self.itemInfo = read_json(str(template_path))["carriages"][str(self.qtrain_info.carriage)]
    #     self.imreader = ImReader(self.qtrain_info.test_train.path, self.qtrain_info.channel, verbose=True, logger=self.logger)
    #     self.splitter = Splitter(self.qtrain_info, self.channel_params.splitter, self.imreader, 
    #                     SETTINGS.TRAIN_LIB_PATH, self.device, axis=self.channel_params.axis, logger=self.logger)
    #     self.locator = Locator(self.qtrain_info, self.channel_params.locator, self.itemInfo, 
    #                             self.device, axis=self.channel_params.axis, logger=self.logger)
    #     self.detector = Detector(self.qtrain_info, self.item_params, self.itemInfo, self.device,
    #                             axis=self.channel_params.axis, logger=self.logger)

    def run2d(self, *args, **kwargs):
        raise NotImplementedError

    def run3d(self, *args, **kwargs):
        raise NotImplementedError

    def run2d_hist(self, *args, **kwargs):
        raise NotImplementedError

    def run3d_hist(self, *args, **kwargs):
        raise NotImplementedError
    
    @TryExcept()
    def run(self):
        raise NotImplementedError

    def _dev_generate_cutpoints_(self, save_path, cutframe_idxes=None, imread=imread_tenth):
        """generate cut points for training"""
        if cutframe_idxes is not None:
            self.splitter.update_cutframe_idx(cutframe_idxes[0], cutframe_idxes[1])
        else:
            cutframe_idxes = self.splitter.get_approximate_cutframe_idxes()
            # update cutframe idx
            self.splitter.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], 
                        cutframe_idxes[self.qtrain_info.carriage])

        self.splitter._dev_generate_cutpoints_img_(save_path, imread=imread_tenth)
    
    def _dev_generate_template_(self, save_path, cutframe_idxes=None, imread=imread_quarter):
        """Generate carriage template"""
        if cutframe_idxes is not None:
            self.splitter.update_cutframe_idx(cutframe_idxes[0], cutframe_idxes[1])
        else:
            cutframe_idxes = self.splitter.get_approximate_cutframe_idxes()
            # update cutframe idx
            self.splitter.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], 
                        cutframe_idxes[self.qtrain_info.carriage])

        self.splitter._dev_generate_car_template_(save_path, imread=imread)

    def _dev_generate_anchors_(self, save_path, cutframe_idxes=None, label_list=[]):
        """Generate anchor image"""
        if cutframe_idxes is not None:
            self.splitter.update_cutframe_idx(cutframe_idxes[0], cutframe_idxes[1])
        else:
            cutframe_idxes = self.splitter.get_approximate_cutframe_idxes()
            # update cutframe idx
            self.splitter.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], 
                        cutframe_idxes[self.qtrain_info.carriage])
        test_startline, test_endline = self.splitter.get_specific_cutpoints()

        self.channel_params =  self.channel_params[str(self.qtrain_info.channel)]
        self.qtrain_info.test_train.startline = test_startline
        self.qtrain_info.test_train.endline = test_endline
        img_h = self.channel_params.img_h 
        img_w = self.channel_params.img_w
        test_img = read_segmented_img(self.imreader, test_startline, test_endline, 
                imread_full, axis=self.channel_params.axis)
        
        # 定位
        template_path = self.template_path / "template.json"
        itemInfo = read_json(str(template_path))["carriages"][str(self.qtrain_info.carriage)]

        self.locator.update_test_traininfo(test_startline, test_endline)
        self.locator.update_temp_traininfo(itemInfo["startline"], itemInfo["endline"])
        anchor_bboxes = json2bbox_formater(itemInfo.get("anchors", []))
        self.locator._dev_generate_anchors_img_(anchor_bboxes, save_path, test_img, img_h, img_w,
                     label_list=label_list)

    def _dev_generate_items_(self, save_path, item_bboxes, test_img, test_startline, img_w, img_h, axis, label_list=[]):
        """Generate item images"""
        import cv2
        save_path.mkdir(exist_ok=True, parents=True)
        
        for bbox in item_bboxes:
            if bbox.name in label_list:
                curr_rect_p = frame2rect(bbox.curr_rect, test_startline, img_h, img_w,
                            start_minor_axis_fp=0, axis=axis)
                curr_rect_img = crop_segmented_rect(test_img, curr_rect_p)

                fname = concat_str(self.qtrain_info.minor_train_code,
                    self.qtrain_info.train_num, self.qtrain_info.train_sn,
                    self.qtrain_info.channel, self.qtrain_info.carriage,
                    bbox.name, bbox.index)
                img_path = save_path / (fname + ".jpg")
                cv2.imwrite(str(img_path), curr_rect_img)
                d_logger.info(f">>> {fname}")

    def _dev_generate_items_json_template_(self, save_path, anchor_bboxes=[], item_bboxes=[], chunk_bboxes=[]):
        """Generate json template"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        json_dict = {}
        json_dict["train_info"] = self.qtrain_info.to_dict()
        json_dict["anchors"] = [i.to_dict() for i in anchor_bboxes] if anchor_bboxes else []
        json_dict["items"] = [i.to_dict() for i in item_bboxes] if item_bboxes else []
        json_dict["chunks"] = [i.to_dict() for i in chunk_bboxes] if chunk_bboxes else []

        fname = concat_str(self.qtrain_info.channel, self.qtrain_info.carriage)
        fname = save_path / (fname + ".json")
        write_json(fname, json_dict)


class CropToolDev:
    """开发阶段使用的生成，或者截图工具汇总"""
    def __init__(self):
        pass
    
    @staticmethod
    def _dev_generate_items(save_path, img,bboxes, startline, img_h, img_w, axis, qtrain_info, 
                        target_item_list=[], rect_type="proposal_rect"):
        if not bboxes:
            return 

        save_path = Path(save_path)
        for bbox in bboxes:
            if bbox.name in target_item_list or len(target_item_list) == 0:
                curr_rect_p = frame2rect(eval(f"bbox.{rect_type}"), startline, img_h, img_w, start_minor_axis_fp=0, axis=axis)
                curr_rect_img = crop_segmented_rect(img, curr_rect_p)
                
                fname = concat_str(qtrain_info.minor_train_code, qtrain_info.train_num, qtrain_info.train_sn,
                        qtrain_info.channel, qtrain_info.carriage, bbox.name, bbox.index)
                img_path = save_path / (fname + ".jpg")
                cv2.imwrite(str(img_path), curr_rect_img)
                print(f">>> {fname}")

    # @staticmethod
    # def _dev_generate_cutpoints(save_path, splitter, qtrain_info, cutframe_idxes=None, imread=imread_tenth):
    #     if cutframe_idxes is not None:
    #         splitter.update_cutframe_idx(cutframe_idxes[0], cutframe_idxes[1])
    #     else:
    #         cutframe_idxes = splitter.get_approximate_cutframe_idxes()
    #         # update cutframe idx
    #         splitter.update_cutframe_idx(cutframe_idxes[qtrain_info.carriage-1], 
    #                     cutframe_idxes[qtrain_info.carriage])

    #     splitter._dev_generate_cutpoints_img_(save_path, imread=imread)








        