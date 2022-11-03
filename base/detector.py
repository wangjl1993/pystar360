from importlib import import_module

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
        self.logger = logger
        self.debug = debug
        self.mac_password = mac_password

    def detect_items(self, item_bboxes, test_img, test_startline, img_w, img_h, **kwargs):
        if not item_bboxes:
            return []
        item_bboxes_dict = bboxes_collector(item_bboxes)

        if self.logger:
            self.logger.info(f">>> Start detection...")
        else:
            d_logger.info(f">>> Start detection...")

        new_item_bboxes = []
        for label_name, item_bboxes_list in item_bboxes_dict.items():
            if self.logger:
                self.logger.info(f">>> Processing items with label: {label_name}")
            else:
                d_logger.info(f">>> Processing items with label: {label_name}")

            try:
                # load params
                item_params = self.item_params[label_name]
            except KeyError:
                if self.logger:
                    self.logger.info(f">>> Params not provided. Skip items with label: {label_name}")
                else:
                    d_logger.info(f">>> Params not provided. Skip items with label: {label_name}")
                # new_item_bboxes.extend(item_bboxes_list)
                continue

            try:
                # load func
                # func_obj = eval(item_params.func)
                func_obj = getattr(import_module(item_params.module), item_params.func)
            except:
                if self.logger:
                    self.logger.info(f">>> Fail to call {item_params.func}")
                else:
                    d_logger.info(f">>> Fail to call {item_params.func}")
                # new_item_bboxes.extend(item_bboxes_list)
                continue

            # init object
            func = func_obj(
                item_params=item_params.params,
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

            # run func
            local_item_bboxes = func(item_bboxes_list, test_img, test_startline, img_h, img_w, **kwargs)

            new_item_bboxes.extend(local_item_bboxes)

        if self.logger:
            self.logger.info(f">>> Finished detection...")
        else:
            d_logger.info(f">>> Finished detection...")

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
