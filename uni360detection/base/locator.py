
import numpy as np
from copy import deepcopy
from uni360detection.utilities.fileManger import LABELME_RECT_TEMPLATE, write_json, LABELME_TEMPLATE
from uni360detection.base.dataStruct import bbox_formater
from uni360detection.utilities.helper import *
from uni360detection.yolo.inference import *

def cal_coord_by_ratio_adjustment(points, temp_startline, temp_carspan, test_startline, test_carspan, axis=1):
    if axis == 1: # horizontal 
        pts1 = [(points[0][0] - temp_startline) / temp_carspan * test_carspan + test_startline, points[0][1]]
        pts2 = [(points[1][0] - temp_startline) / temp_carspan * test_carspan + test_startline, points[1][1]]
        new_points = [pts1, pts2]
        # new_points = [[(pt[0] - temp_startline) / temp_carspan * test_carspan + test_startline, pt[1]] for pt in points]
    elif axis == 0: # vertical image
        pts1 = [points[0][0], (points[0][1] - temp_startline) / temp_carspan * test_carspan + test_startline]
        pts2 = [points[1][0], (points[1][1] - temp_startline) / temp_carspan * test_carspan + test_startline]
        new_points = [pts1, pts2]
        # new_points = [[pt[0], (pt[1] - temp_startline) / temp_carspan * test_carspan + test_startline] for pt in points]
    else:
        raise ValueError("Please provide valid axis 0 or 1")
    return new_points

def cal_new_pts(pt, temp_pts, first_ref, ref_segl, first_cur, cur_segl, main_axis, minor_axis_poly_func):
    new_pt = (pt - first_ref) / ref_segl * cur_segl + first_cur
    points = temp_pts
    if main_axis == 1:
        proposal_pts = [min(max(minor_axis_poly_func(points[0]), 0), 1), new_pt]
    else:
        proposal_pts = [new_pt, min(max(minor_axis_poly_func(points[1]), 0), 1)]
    return proposal_pts

class Locator:
    def __init__(self,  qtrain_info, local_params, itemInfo, device, axis=1, logger=None):
        
        # query information 
        self.qtrain_info = qtrain_info

        # local params 
        self.local_params = local_params
        
        # template information 
        self.itemInfo = itemInfo 
        self.temp_startline = self.itemInfo["startline"]
        self.temp_endline = self.itemInfo["endline"]
        self.temp_carspan = self.temp_endline - self.temp_startline

        # device 
        self.device = device 

        self.axis = axis

        # logger 
        self.logger = logger 
        self.new_anchor_bboxes = [] 

    def update_test_traininfo(self, test_startline, test_endline):
        self.test_startline = test_startline
        self.test_endline = test_endline  
        self.test_carspan = abs(self.test_endline - self.test_startline)

    def locate_anchors_yolo(self, test_img, img_h, img_w):
        anchor_bboxes = bbox_formater(self.itemInfo["anchors"])

        # load model 
        model = YoloInfer(self.local_params.model_path, 
                self.device, imgsz=self.local_params.imgsz,
                logger=self.logger)
        label_translator = self.local_params.label_translator

        self.new_anchor_bboxes = [] 
        for idx, bbox in enumerate(anchor_bboxes):
            # get proposal rect in frame points
            bbox.proposal_rect = cal_coord_by_ratio_adjustment(bbox.temp_rect, self.temp_startline, self.temp_carspan, 
            self.test_startline, self.test_carspan, self.axis)

            # proposal rect from frame points to local pixel values 
            propoal_rect_p = frame2rect(bbox.proposal_rect, self.test_startline, 
                            img_h, img_w, start_minor_axis_fp=0, axis=self.axis)
            # proposal rect image 
            proposal_rect_img = crop_segmented_rect(test_img, propoal_rect_p)
            proposal_rect_img = cv2.cvtColor(proposal_rect_img, cv2.COLOR_GRAY2BGR)

            # infer
            temp_outputs = model.infer(proposal_rect_img, conf_thres=self.local_params.conf_thres,
                        iou_thres=self.local_params.iou_thres)
            temp_outputs = [i for i in temp_outputs if label_translator[int(i[0])] == bbox.name]
            
            # 0:class, 1:ctrx, 2:ctry, 3;w, 4:h, 5:confidence,
            if len(temp_outputs) > 0:
                # max_output = max(temp_outputs, key=lambda x: x[0])
                max_output = select_best_yolobox(temp_outputs, self.local_params.method)
                # [sx, sy] = bbox.proposal_rect
                # ref_w = bbox.proposal_rect[1][0] - bbox.proposal_rect[0][0]
                # ref_h = bbox.proposal_rect[1][1] - bbox.proposal_rect[0][1]
                # local rect to global rect 
                # bbox.curr_rect = yolo_xywh2xyxy(max_output[1:], sx, sy, ref_h, ref_w)
                bbox.curr_rect = yolo_xywh2xyxy_v2(max_output[1:5], bbox.proposal_rect)
                bbox.score = max_output[0]
                bbox.conf_thres = self.local_params.conf_thres
                self.new_anchor_bboxes.append(bbox)


    def _update_minor_axis_affine_transform_matrix(self, minor_axis_affine_maxtrix):
        """ minor_axis_affine_maxtrix is a list, for linear appro kx + b = y, then z = [k, b]"""
        try:
            minor_axis_poly_func = np.poly1d(minor_axis_affine_maxtrix)
        except:
            if self.logger:
                self.logger.warning(f"Minor axis affine transform matrix is invalid: {minor_axis_affine_maxtrix}")
            raise ValueError(f"Minor axis affine transform matrix is invalid: {minor_axis_affine_maxtrix}")
        return minor_axis_poly_func

    
    def _auto_update_minor_axis_affine_transform_matrix(self, x, y, poly_order=2):
        if not isinstance(x, list) or not isinstance(y, list):
            x = np.array(x)
            y = np.array(y)


        z = np.polyfit(x, y, poly_order) 
        minor_axis_poly_func = np.poly1d(z)
        return minor_axis_poly_func

                
    def locate_bboxes_according2anchors(self, item_bboxes):
        if not item_bboxes:
            return []
        anchor_bboxes = self.new_anchor_bboxes

        # if axis == 0, vertical, main_axis = y(index=1) in [x, y], minor_axis = x(index=0)
        # if axis == 1, horizontal, main_axis = x(index=0) in [x, y], minor_axis = y(index=1)
        main_axis = 1 if self.axis == 0 else 0 
        minor_axis = self.axis 

        temp_anchor_points, curr_anchor_points = [], []
        for anchor in anchor_bboxes:
            temp_anchor_points.append(anchor.orig_rect[0]) # left top pt [x, y]
            temp_anchor_points.append(anchor.orig_rect[1]) # right bottom pt [x, y]
            curr_anchor_points.append(anchor.curr_rect[0]) # left top pt [x, y]
            curr_anchor_points.append(anchor.curr_rect[1]) # right bottom pt [x, y]
        temp_anchor_points = sorted(temp_anchor_points, key=lambda a: a[main_axis])
        curr_anchor_points = sorted(curr_anchor_points, key=lambda a: a[main_axis])

        # calculate minor axis shift using ax + b = y
        if self.local_params.minor_axis_affine_maxtrix:
            minor_axis_poly_func = self._update_minor_axis_affine_transform_matrix(self.local_params.minor_axis_affine_maxtrix)
        elif self.local_params.auto_minor_axis_adjust:
            minor_axis_poly_func = self._auto_update_minor_axis_affine_transform_matrix([pt[minor_axis] for pt in temp_anchor_points],
                                                    [pt[minor_axis] for pt in curr_anchor_points])
        else:
            raise NotImplementedError

        # process main axis
        # add startline point and endline point (+ 2 - 1 = + 1)
        # number of segments 
        temp_anchor_points = [pt[main_axis] for pt in temp_anchor_points]
        curr_anchor_points = [pt[main_axis] for pt in curr_anchor_points]
        seg_cnt = len(temp_anchor_points) + 1
        seg_cnt2 = len(curr_anchor_points) + 1
        assert seg_cnt == seg_cnt2

        for i in range(seg_cnt):
            # calculate reference span and test span
            if i == 0:
                first_ref = self.temp_startline
                second_ref = temp_anchor_points[i]
                first_cur = self.test_startline
                second_cur = curr_anchor_points[i]
            elif i == seg_cnt - 1:
                first_ref = temp_anchor_points[i - 1]
                second_ref = self.temp_endline
                first_cur = curr_anchor_points[i - 1]
                second_cur = self.test_endline
            else:
                first_ref = temp_anchor_points[i - 1]
                second_ref = temp_anchor_points[i]
                first_cur = curr_anchor_points[i - 1]
                second_cur = curr_anchor_points[i]

            ref_segl = second_ref - first_ref
            cur_segl = second_cur - first_cur

            # for each template interval, calculate the ratio difference
            # rescale the current interval
            for _, bbox in enumerate(item_bboxes):
                pt0 = max(bbox.orig_rect[0][main_axis], self.temp_startline)
                pt1 = min(bbox.orig_rect[1][main_axis], self.temp_endline)

                if first_ref <= pt0 <= second_ref:
                    bbox.proposal_rect[0] = cal_new_pts(pt0, bbox.orig_rect[0], first_ref, ref_segl,
                                            first_cur, cur_segl, main_axis, minor_axis_poly_func)
                    bbox.curr_rect[0] = bbox.proposal_rect[0].to_list() # copy

                if first_ref <= pt1 <= second_ref:
                    bbox.proposal_rect[1] = cal_new_pts(pt1, bbox.orig_rect[1], first_ref, ref_segl,
                                            first_cur, cur_segl, main_axis, minor_axis_poly_func)
                    bbox.curr_rect[1] = bbox.proposal_rect[1].to_list() # copy

        return item_bboxes


    def _dev_generate_bboxes_img_(self, save_path, test_img, img_h, img_w, box_type="anchors"):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        print(">>> Start cropping...")
        anchor_bboxes = bbox_formater(self.itemInfo[box_type])
        for idx, bbox in enumerate(anchor_bboxes):
            new_template = deepcopy(LABELME_TEMPLATE)
            new_rectangle = deepcopy(LABELME_RECT_TEMPLATE)

            # get proposal rect in frame points
            proposal_rect = cal_coord_by_ratio_adjustment(bbox.temp_rect, self.temp_startline, self.temp_carspan, 
                                self.test_startline, self.test_carspan, self.axis)
            curr_rect = cal_coord_by_ratio_adjustment(bbox.orig_rect, self.temp_startline, self.temp_carspan, 
                                self.test_startline, self.test_carspan, self.axis)
            
            # proposal rect from frame points to local pixel values 
            proposal_rect_p = frame2rect(proposal_rect, self.test_startline, 
                            img_h, img_w, start_minor_axis_fp=0, axis=self.axis)
            curr_rect_p = frame2rect(curr_rect, self.test_startline, 
                            img_h, img_w, start_minor_axis_fp=0, axis=self.axis)

            _, _, imageWidth, imageHeight = xyxy2left_xywh(proposal_rect_p)
            new_rectangle["points"] = xyxy_nested(curr_rect_p, proposal_rect_p)
            new_rectangle["label"] = bbox.label

            # proposal rect image 
            proposal_rect_img = crop_segmented_rect(test_img, proposal_rect_p)
            
            fname = "{}_{}_{}_{}_{}_{}_{}_{}".format(
                box_type, self.qtrain_info.minor_train_code, self.qtrain_info.train_num, 
                self.qtrain_info.train_sn, self.qtrain_info.channel, self.qtrain_info.carriage, 
                bbox.name, idx)
            new_template["shapes"].append(new_rectangle)
            new_template["imageHeight"] = int(imageHeight)
            new_template["imageWidth"] = int(imageWidth)
            new_template["imagePath"] = fname + ".jpg"
            img_fname = save_path / (fname + ".jpg")
            cv2.imwrite(str(img_fname), proposal_rect_img)
            json_fname = save_path / (fname + ".json")
            write_json(str(json_fname), new_template)
            print(f">>> {fname}.")



