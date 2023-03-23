from importlib import import_module
from typing import Dict, List
from omegaconf import DictConfig, ListConfig

from pystar360.base.dataStruct import bboxes_collector
from pystar360.algo.algoBase import NullDetection
from pystar360.utilities._logger import d_logger

__all__ = ["Detector"]


class Detector:
    def __init__(self, qtrain_info, item_params, device, axis=1, logger=None, debug=False, mac_password=None):
        # query train informaiton
        self.qtrain_info = qtrain_info

        # params
        self.item_params = item_params

        # itemInfo
        self.device = device
        self.axis = axis
        self.logger = logger if logger else d_logger
        self.debug = debug
        self.mac_password = mac_password

    def detect_items(self, item_bboxes, test_img, test_startline, img_w, img_h, **kwargs):
        if not item_bboxes:
            return []

        # collect bboxes
        item_bboxes_dict = bboxes_collector(item_bboxes)
        self.logger.info(f">>> Start detection...")

        new_item_bboxes = []
        for label_name, item_bboxes_list in item_bboxes_dict.items():
            self.logger.info(f">>> Processing items with label: {label_name}")

            # load params
            try:
                item_params = self.item_params[label_name]
            except KeyError:
                self.logger.info(f">>> Params not provided. Skip items with label: {label_name}")
                # new_item_bboxes.extend(item_bboxes_list)
                continue

            # check pipeline count
            if isinstance(item_params, (Dict, DictConfig)):
                item_params = [item_params]
            elif isinstance(item_params, (List, ListConfig)):
                pass
            else:
                raise ValueError(
                    f">>> Item params {type(item_params)} for {label_name} is invalid. Please check your configure."
                )

            # start looping
            for idx, item_param in enumerate(item_params, start=1):
                self.logger.info(f">>> Call method {idx} {item_param.func} for {label_name}.")

                if self.debug:  # load func
                    func_obj = getattr(import_module(item_param.module), item_param.func)
                else:
                    try:
                        func_obj = getattr(import_module(item_param.module), item_param.func)
                    except:
                        self.logger.info(f">>> Fail to call {item_param.func}")
                        # new_item_bboxes.extend(item_bboxes_list)
                        continue

                # init object
                func = func_obj(
                    item_params=item_param.params,
                    device=self.device,
                    logger=self.logger,
                    axis=self.axis,
                    mac_password=self.mac_password,
                    debug=self.debug,
                )

                # sorting
                if self.axis == 0:
                    item_bboxes_list = sorted(item_bboxes_list, key=lambda x: x.proposal_rect[0][1])
                else:
                    item_bboxes_list = sorted(item_bboxes_list, key=lambda x: x.proposal_rect[0][0])

                # run algorithm
                if self.debug:
                    local_item_bboxes = func(item_bboxes_list, test_img, test_startline, img_h, img_w, **kwargs)
                    item_bboxes_list = local_item_bboxes
                else:
                    try:
                        local_item_bboxes = func(item_bboxes_list, test_img, test_startline, img_h, img_w, **kwargs)
                        item_bboxes_list = local_item_bboxes
                    except Exception as e:
                        local_item_bboxes = item_bboxes_list
                        self.logger.info(str(e))

            new_item_bboxes.extend(local_item_bboxes)

        self.logger.info(f">>> Finished detection...")
        return new_item_bboxes

    def _dev_item_null_detection_(self, item_bboxes, *args, **kwargs):
        if not item_bboxes:
            return []

        item_bboxes_dict = bboxes_collector(item_bboxes)

        new_item_bboxes = []
        for _, item_bboxes_list in item_bboxes_dict.items():

            # sorting
            if self.axis == 0:
                item_bboxes_list = sorted(item_bboxes_list, key=lambda x: x.proposal_rect[0][1])
            else:
                item_bboxes_list = sorted(item_bboxes_list, key=lambda x: x.proposal_rect[0][0])

            # run func
            func = NullDetection()
            local_item_bboxes = func(item_bboxes_list)
            new_item_bboxes.extend(local_item_bboxes)

        return new_item_bboxes
