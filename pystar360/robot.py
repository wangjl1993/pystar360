
from pathlib import Path
from pystar360.utilities.fileManger import  read_yaml, write_json
from pystar360.utilities.helper import concat_str, read_segmented_img, imread_full
from pystar360.base.dataStruct import json2bbox_formater



class pyStar360RobotBase:
    def __init__(self, qtrain_info, channel_param, item_params, template_path, device="cuda:0", logger=None):
        self.qtrain_info = qtrain_info 
        self.channel_params = read_yaml(channel_param)[str(qtrain_info.channel)]
        self.item_params = read_yaml(item_params)
        self.template_path = Path(template_path) / str(qtrain_info.channel)
        self.device = device 
        self.logger = logger 

    ###  example
    # def __post_init(self):
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

    def run(self):
        raise NotImplementedError

    def _dev_generate_cutpoints_(self, save_path, cutframe_idxes=None):
        # ---------- generate cut points for training 
        if cutframe_idxes is not None:
            self.splitter.update_cutframe_idx(cutframe_idxes[0], cutframe_idxes[1])
        else:
            cutframe_idxes = self.splitter.get_approximate_cutframe_idxes()
            # update cutframe idx
            self.splitter.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], 
                        cutframe_idxes[self.qtrain_info.carriage])

        self.splitter._dev_generate_cutpoints_img_(save_path)
    
    def _dev_generate_template_(self, save_path, cutframe_idxes=None):
        if cutframe_idxes is not None:
            self.splitter.update_cutframe_idx(cutframe_idxes[0], cutframe_idxes[1])
        else:
            cutframe_idxes = self.splitter.get_approximate_cutframe_idxes()
            # update cutframe idx
            self.splitter.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], 
                        cutframe_idxes[self.qtrain_info.carriage])

        self.splitter._dev_generate_car_template_(save_path)

    def _dev_generate_anchors_(self, save_path, cutframe_idxes=None):
        if cutframe_idxes is not None:
            self.splitter.update_cutframe_idx(cutframe_idxes[0], cutframe_idxes[1])
        else:
            cutframe_idxes = self.splitter.get_approximate_cutframe_idxes()
            # update cutframe idx
            self.splitter.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], 
                        cutframe_idxes[self.qtrain_info.carriage])
        test_startline, test_endline = self.splitter.get_specific_cutpoints()

        self.qtrain_info.test_train.startline = test_startline
        self.qtrain_info.test_train.endline = test_endline
        img_h = self.channel_params.img_h 
        img_w = self.channel_params.img_w
        test_img = read_segmented_img(self.imreader, test_startline, test_endline, 
                imread_full, axis=self.channel_params.axis)
        
        # 定位
        self.locator.update_test_traininfo(test_startline, test_endline)
        self.locator._dev_generate_anchors_img_(save_path, test_img, img_h, img_w)
    
    def _dev_generate_items_template_(self, save_path, cutframe_idxes=None):
        if cutframe_idxes is not None:
            self.splitter.update_cutframe_idx(cutframe_idxes[0], cutframe_idxes[1])
        else:
            cutframe_idxes = self.splitter.get_approximate_cutframe_idxes()
            # update cutframe idx
            self.splitter.update_cutframe_idx(cutframe_idxes[self.qtrain_info.carriage-1], 
                        cutframe_idxes[self.qtrain_info.carriage])
        test_startline, test_endline = self.splitter.get_specific_cutpoints()

        self.qtrain_info.test_train.startline = test_startline
        self.qtrain_info.test_train.endline = test_endline
        img_h = self.channel_params.img_h 
        img_w = self.channel_params.img_w
        test_img = read_segmented_img(self.imreader, test_startline, test_endline, 
                imread_full, axis=self.channel_params.axis)
        
        # locate
        self.locator.update_test_traininfo(test_startline, test_endline)
        anchor_bboxes = json2bbox_formater(self.itemInfo.get("anchors", []))
        anchor_bboxes = self.locator.locate_anchors_yolo(anchor_bboxes, test_img, img_h, img_w)
        self.locator.get_affine_transformation(anchor_bboxes) # template VS test anchors relationship 

        item_bboxes = json2bbox_formater(self.itemInfo.get("items", []))
        item_bboxes = self.locator.locate_bboxes_according2anchors(item_bboxes)
        chunk_bboxes = json2bbox_formater(self.itemInfo.get("chunks", []))
        chunk_bboxes = self.locator.locate_bboxes_according2anchors(chunk_bboxes)

        item_bboxes = self.detector._dev_item_null_detection_(item_bboxes)

        json_dict = {}

        json_dict["train_info"] = self.qtrain_info.to_dict()
        json_dict["anchors"] = [i.to_dict() for i in anchor_bboxes]
        json_dict["items"] = [i.to_dict() for i in item_bboxes]
        json_dict["chunks"] = [i.to_dict() for i in chunk_bboxes]

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        fname = concat_str(self.qtrain_info.channel, self.qtrain_info.carriage)
        fname = save_path / (fname + ".json")
        write_json(fname, json_dict)









        