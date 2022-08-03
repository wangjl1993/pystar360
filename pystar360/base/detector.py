from pystar360.algo.algoBase import NullDetection
from pystar360.algo.defectType1 import *
from pystar360.algo.defectType2 import *

# 所有的检测方程都需要import, 为了能成功eval func 如 from xxx_algos import * 

def bboxes_collector(bboxes):
    """collect items according to its label"""
    output = {}
    for b in bboxes:
        if b.name in output:
            output[b.name].append(b)
        else:
            output[b.name] = [b]
    return output 


class Detector:
    def __init__(self, qtrain_info, item_params, itemInfo, device, axis=1, logger=None):
        # query train informaiton 
        self.qtrain_info = qtrain_info

        # params 
        self.item_params = item_params 

        # itemInfo 
        self.itemInfo = itemInfo
        self.device = device
        self.axis = axis 
        self.logger = logger 

    def detect_items(self, item_bboxes, test_img, test_startline, img_w, img_h):
        if not item_bboxes:
            return [] 
        item_bboxes_dict = bboxes_collector(item_bboxes)

        if self.logger:
            self.logger.info(f">>> Start detection...")
        else:
            print(f">>> Start detection...")

        new_item_bboxes = []
        for label_name, item_bboxes_list in item_bboxes_dict.items():
            if self.logger:
                self.logger.info(f">>> Processing items with label: {label_name}")
            else:
                print(f">>> Processing items with label: {label_name}")


            try:
                # load params 
                item_params = self.item_params[label_name]
            except KeyError:
                if self.logger:
                    self.logger.info(f">>> Params not provided. Skip items with label: {label_name}")
                else:
                    print(f">>> Params not provided. Skip items with label: {label_name}")
                new_item_bboxes.extend(item_bboxes_list)
                continue 

            try:
                # load func 
                func_obj = eval(item_params.func)
            except NameError:
                if self.logger:
                    self.logger.info(f">>> Fail to call {item_params.func}")
                else:
                    print(f">>> Fail to call {item_params.func}")
                new_item_bboxes.extend(item_bboxes_list)
                continue 
            
            # init object 
            func = func_obj(item_params = item_params.params,
                            device = self.device,
                            logger = self.logger)
             # sorting
            if self.axis == 0:
                item_bboxes_list = sorted(item_bboxes_list, key=lambda x: x.proposal_rect[0][1])
            else:
                item_bboxes_list = sorted(item_bboxes_list, key=lambda x: x.proposal_rect[0][0])
            
            # run func
            local_item_bboxes = func(item_bboxes_list, test_img, test_startline, img_h, img_w)

            new_item_bboxes.extend(local_item_bboxes)
        
        
        if self.logger:
            self.logger.info(f">>> Finished detection...")
        else:
            print(f">>> Finished detection...")
            
        return new_item_bboxes


    def _dev_item_null_detection_(self, item_bboxes):
        if not item_bboxes:
            return [] 

        item_bboxes_dict = bboxes_collector(item_bboxes)

        new_item_bboxes = []
        for _, item_bboxes_list in item_bboxes_dict.items():
            
            func = NullDetection()
            # sorting
            if self.axis == 0:
                item_bboxes_list = sorted(item_bboxes_list, key=lambda x: x.proposal_rect[0][1])
            else:
                item_bboxes_list = sorted(item_bboxes_list, key=lambda x: x.proposal_rect[0][0])
            
            # run func
            local_item_bboxes = func(item_bboxes_list)
            new_item_bboxes.extend(local_item_bboxes)

        return new_item_bboxes 
