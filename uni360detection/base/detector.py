


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

    # def update_test_img_info(self, img, startline, img_w, img_h):
    #     self.test_img = img
    #     self.test_startline = startline
    #     self.img_w = img_w
    #     self.img_h = img_h

    def detect_items(self, item_bboxes, test_img, test_startline, img_w, img_h):
        item_bboxes_dict = bboxes_collector(item_bboxes)

        if self.logger:
            self.logger.info(f">>> Start detection...")
        else:
            print(f">>> Start detection...")

        new_item_bboxes = []
        for label_name, item_bboxes_list in item_bboxes_dict.item():
            if self.logger:
                self.logger.info(f">>> Processing items with label: {label_name}")

                try:
                    item_params = self.item_params[label_name]
                except KeyError:
                    if self.logger:
                        self.logger.info(f">>> Skip items with label: {label_name}")
                    else:
                        print(f">>> Skip items with label: {label_name}")
                    continue 

                try:
                    func_obj = eval(item_params.func)
                except NameError:
                    if self.logger:
                        self.logger.info(f">>> Fail to call {item_params.func}")
                    else:
                        print(f">>> Fail to call {item_params.func}")
                    continue 

                func = func_obj(item_bboxes_list=item_bboxes_list, 
                                device = self.device,
                                logger = self.logger,
                                item_params = item_params.params)
                # run func
                local_item_bboxes = func(test_img = test_img, 
                                         test_startline = test_startline,
                                         img_h = img_h, 
                                         img_w = img_w)

        new_item_bboxes.extend(local_item_bboxes)
        if self.logger:
            self.logger.info(f">>> Finished detection...")
        else:
            print(f">>> Finished detection...")
            
        return new_item_bboxes

                
