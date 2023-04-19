import numpy as np
import cv2

from functools import partial
from copy import deepcopy
from pathlib import Path
from pystar360.base.dataStruct import BBox, QTrainInfo, json2bbox_formater
from pystar360.utilities.fileManger import write_json, LABELME_TEMPLATE, LABELME_RECT_TEMPLATE
from pystar360.yolo.inference import YoloInfer, yolo_xywh2xyxy_v2, select_best_yolobox
from pystar360.cls.inference import ClsInfer
from pystar360.utilities.helper import (
    frame2rect, crop_segmented_rect, 
    get_label_num2check, xyxy2left_xywh, xyxy_nested, 
    resize_img, concat_str, hungary_match
)
from pystar360.utilities.logger import w_logger
from pystar360.utilities.helper3d import map3d_for_bboxes
from typing import (
    Union, List, Dict,
    Tuple, Optional
)
import itertools

__all__ = ["Locator"]


DEFAULT_MINOR_AXIS_AFFINE_MATRIX = None
DEFAULT_AUTO_MINOR_AXIS_ADJUST = True
DEFAULT_POLY_ORDER = 1


def cal_coord_by_ratio_adjustment(points, temp_startline, temp_carspan, test_startline, test_carspan, axis=1):
    if axis == 1:  # horizontal
        pts1 = [(points[0][0] - temp_startline) / temp_carspan * test_carspan + test_startline, points[0][1]]
        pts2 = [(points[1][0] - temp_startline) / temp_carspan * test_carspan + test_startline, points[1][1]]
        new_points = [pts1, pts2]
        # new_points = [[(pt[0] - temp_startline) / temp_carspan * test_carspan + test_startline, pt[1]] for pt in points]
    elif axis == 0:  # vertical image
        pts1 = [points[0][0], (points[0][1] - temp_startline) / temp_carspan * test_carspan + test_startline]
        pts2 = [points[1][0], (points[1][1] - temp_startline) / temp_carspan * test_carspan + test_startline]
        new_points = [pts1, pts2]
        # new_points = [[pt[0], (pt[1] - temp_startline) / temp_carspan * test_carspan + test_startline] for pt in points]
    else:
        raise ValueError("Please provide valid axis 0 or 1")
    return new_points


def cal_new_pt_in_main_axis(pt, first_ref, ref_segl, first_cur, cur_segl):
    new_pt = (pt - first_ref) / ref_segl * cur_segl + first_cur
    return new_pt


def cal_new_pt_in_minor_axis(pt, minor_axis_poly_func):
    new_pt = min(max(minor_axis_poly_func(pt), 0), 1)
    return new_pt


def cal_new_pts(pt, temp_pts, first_ref, ref_segl, first_cur, cur_segl, main_axis, minor_axis_poly_func):
    new_pt = cal_new_pt_in_main_axis(pt, first_ref, ref_segl, first_cur, cur_segl)

    points = temp_pts
    if main_axis == 1:
        proposal_pts = [cal_new_pt_in_minor_axis(points[0], minor_axis_poly_func), new_pt]
    else:
        proposal_pts = [new_pt, cal_new_pt_in_minor_axis(points[1], minor_axis_poly_func)]
    return proposal_pts


def cal_new_pts_only_minor(temp_pts, main_axis, minor_axis_poly_func):
    points = temp_pts
    if main_axis == 1:
        proposal_pts = [cal_new_pt_in_minor_axis(points[0], minor_axis_poly_func), points[1]]
    else:
        proposal_pts = [points[0], cal_new_pt_in_minor_axis(points[1], minor_axis_poly_func)]
    return proposal_pts


def trans_coords_from_chunk2frame(chunk_rect: list, item_rect: list):
    """translate item coords from ref-chunk to ref-frame

    Args:
        chunk_rect (list): chunk coords (ref-frame)
        item_rect (list): item coords (ref-chunk)

    Returns:
        _type_: item coords (ref-frame)
    """
    chunk_pt0, chunk_pt1 = chunk_rect
    X0, Y0 = chunk_pt0
    X1, Y1 = chunk_pt1
    chunk_h, chunk_w = Y1 - Y0, X1 - X0

    item_pt0, item_pt1 = item_rect
    x0, y0 = item_pt0
    x1, y1 = item_pt1

    new_x0 = (x0 * chunk_w) + X0
    new_y0 = (y0 * chunk_h) + Y0
    new_x1 = (x1 * chunk_w) + X0
    new_y1 = (y1 * chunk_h) + Y0
    return [[new_x0, new_y0], [new_x1, new_y1]]


class Locator:
    def __init__(self, qtrain_info: QTrainInfo, local_params, device, axis=1, debug=False, mac_password=None):
        # query information
        self.qtrain_info = qtrain_info
        # local params
        self.local_params = local_params
        # device
        self.device = device
        # axis
        self.axis = axis
        # if axis == 0, vertical, main_axis = y(index=1) in [x, y], minor_axis = x(index=0)
        # if axis == 1, horizontal, main_axis = x(index=0) in [x, y], minor_axis = y(index=1)
        self.main_axis = 1 if self.axis == 0 else 0
        self.minor_axis = self.axis

        self.debug = debug
        self.mac_password = mac_password

    def update_test_traininfo(self, test_startline, test_endline):
        # test train information
        self.test_startline = test_startline
        self.test_endline = test_endline
        self.test_carspan = abs(self.test_endline - self.test_startline)

    def update_temp_traininfo(self, temp_startline, temp_endline):
        # template(reference) train information
        self.temp_startline = temp_startline
        self.temp_endline = temp_endline
        self.temp_carspan = abs(self.temp_endline - self.temp_startline)

    def locate_anchors_yolo(self, anchor_bboxes, test_img, img_h, img_w, use_ratio_adjust=True, **kwargs):
        # kwargs can be overwritten, so that you can include more anchors model
        # anchor_bboxes = bbox_formater(self.itemInfo["anchors"])
        model_path = kwargs.get("model_path", self.local_params.model_path)
        imgsz = kwargs.get("imgsz", self.local_params.imgsz)
        label_translator = kwargs.get("label_translator", self.local_params.label_translator)
        method = kwargs.get("method", self.local_params.method)
        conf_thres = kwargs.get("conf_thres", self.local_params.conf_thres)
        iou_thres = kwargs.get("iou_thres", self.local_params.iou_thres)

        # load model
        model = YoloInfer(model_path, self.device, imgsz=imgsz, mac_password=self.mac_password)

        new_anchor_bboxes = []
        for _, bbox in enumerate(anchor_bboxes):
            # get proposal rect in frame points
            if use_ratio_adjust:
                bbox.curr_proposal_rect = cal_coord_by_ratio_adjustment(
                    bbox.temp_rect,
                    self.temp_startline,
                    self.temp_carspan,
                    self.test_startline,
                    self.test_carspan,
                    self.axis,
                )

            # proposal rect from frame points to local pixel values
            propoal_rect_p = frame2rect(
                bbox.curr_proposal_rect, self.test_startline, img_h, img_w, start_minor_axis_fp=0, axis=self.axis
            )
            # proposal rect image
            proposal_rect_img = crop_segmented_rect(test_img, propoal_rect_p)
            proposal_rect_img = cv2.cvtColor(proposal_rect_img, cv2.COLOR_GRAY2BGR)

            # infer
            temp_outputs = model.infer(proposal_rect_img, conf_thres=conf_thres, iou_thres=iou_thres)
            temp_outputs = [i for i in temp_outputs if label_translator[int(i[0])] == bbox.name]

            # 0:class, 1:ctrx, 2:ctry, 3;w, 4:h, 5:confidence,
            if len(temp_outputs) > 0:
                max_output = select_best_yolobox(temp_outputs, method)
                bbox.curr_rect = yolo_xywh2xyxy_v2(max_output[1:5], bbox.curr_proposal_rect)
                bbox.conf_score = max_output[-1]
                bbox.conf_thres = conf_thres
                new_anchor_bboxes.append(bbox)

        if self.debug:
            w_logger.info(f">>> Num of anchors before processing: {len(anchor_bboxes)}")
            w_logger.info(f">>> Num of anchors after processing: {len(new_anchor_bboxes)}")

        return new_anchor_bboxes

    def _update_minor_axis_affine_transform_matrix(self, minor_axis_affine_maxtrix):
        """minor_axis_affine_maxtrix is a list, for linear appro kx + b = y, then z = [k, b]"""
        try:
            minor_axis_poly_func = np.poly1d(minor_axis_affine_maxtrix)
        except:
            w_logger.error(f"Minor axis affine transform matrix is invalid: {minor_axis_affine_maxtrix}")
            raise ValueError(f"Minor axis affine transform matrix is invalid: {minor_axis_affine_maxtrix}")
        return minor_axis_poly_func

    def _auto_update_minor_axis_affine_transform_matrix(self, x, y, poly_order=1):
        if not isinstance(x, list) or not isinstance(y, list):
            x = np.array(x)
            y = np.array(y)

        z = np.polyfit(x, y, poly_order)
        minor_axis_poly_func = np.poly1d(z)
        return minor_axis_poly_func

    def get_affine_transformation(self, anchor_bboxes):

        minor_axis_affine_matrix = self.local_params.get("minor_axis_affine_matrix", DEFAULT_MINOR_AXIS_AFFINE_MATRIX)
        auto_minor_axis_adjust = self.local_params.get("auto_minor_axis_adjust", DEFAULT_AUTO_MINOR_AXIS_ADJUST)
        poly_order = self.local_params.get("poly_order", DEFAULT_POLY_ORDER)
        # if not anchor boxes provided
        if len(anchor_bboxes) < 1:
            # process main axis, add startline point and endline point, number of segments (+ 2 - 1 = + 1)
            self.temp_anchor_points = [self.temp_startline, self.temp_endline]
            self.curr_anchor_points = [self.test_startline, self.test_endline]

            # calculate minor axis shift using ax + b = y
            if minor_axis_affine_matrix:
                self.minor_axis_poly_func = self._update_minor_axis_affine_transform_matrix(minor_axis_affine_matrix)
            else:
                # y = 1*x + 0
                self.minor_axis_poly_func = self._update_minor_axis_affine_transform_matrix([1, 0])
        else:
            # get template anchors and test anchors points
            self.temp_anchor_points, self.curr_anchor_points = [], []
            for anchor in anchor_bboxes:
                self.temp_anchor_points.append(anchor.orig_rect[0])  # left top pt [x, y]
                self.temp_anchor_points.append(anchor.orig_rect[1])  # right bottom pt [x, y]
                self.curr_anchor_points.append(anchor.curr_rect[0])  # left top pt [x, y]
                self.curr_anchor_points.append(anchor.curr_rect[1])  # right bottom pt [x, y]

            # sorting
            self.temp_anchor_points = sorted(self.temp_anchor_points, key=lambda a: a[self.main_axis])
            self.curr_anchor_points = sorted(self.curr_anchor_points, key=lambda a: a[self.main_axis])

            # calculate minor axis shift using ax + b = y
            if minor_axis_affine_matrix:
                self.minor_axis_poly_func = self._update_minor_axis_affine_transform_matrix(minor_axis_affine_matrix)
            elif auto_minor_axis_adjust:
                variable_x = [pt[self.minor_axis] for pt in self.temp_anchor_points]
                variable_y = [pt[self.minor_axis] for pt in self.curr_anchor_points]
                self.minor_axis_poly_func = self._auto_update_minor_axis_affine_transform_matrix(
                    variable_x, variable_y, poly_order=poly_order
                )
            else:
                raise NotImplementedError

            self.temp_anchor_points = [pt[self.main_axis] for pt in self.temp_anchor_points]
            self.curr_anchor_points = [pt[self.main_axis] for pt in self.curr_anchor_points]
            self.temp_anchor_points = [self.temp_startline] + self.temp_anchor_points + [self.temp_endline]
            self.curr_anchor_points = [self.test_startline] + self.curr_anchor_points + [self.test_endline]

        if self.debug:
            w_logger.info(
                f">>> Minor axis adjustment [linear transformation]: {self.minor_axis_poly_func.coefficients}"
            )
            w_logger.info(f">>> Main axis adjustment [ratio adjustment]: {len(self.temp_anchor_points)} points")

    def locate_bboxes_according2anchors(self, bboxes):
        if not bboxes:
            return []

        temp_anchor_points = self.temp_anchor_points
        curr_anchor_points = self.curr_anchor_points
        seg_cnt = len(self.temp_anchor_points) - 1
        seg_cnt2 = len(self.curr_anchor_points) - 1
        assert seg_cnt == seg_cnt2

        for i in range(seg_cnt):
            first_ref = temp_anchor_points[i]
            second_ref = temp_anchor_points[i + 1]
            first_cur = curr_anchor_points[i]
            second_cur = curr_anchor_points[i + 1]

            # segmentation length
            ref_segl = second_ref - first_ref
            cur_segl = second_cur - first_cur

            if ref_segl == 0 or cur_segl == 0:
                continue

            # partial function
            cal_new_pts_partial = partial(
                cal_new_pts,
                first_ref=first_ref,
                ref_segl=ref_segl,
                first_cur=first_cur,
                cur_segl=cur_segl,
                main_axis=self.main_axis,
                minor_axis_poly_func=self.minor_axis_poly_func,
            )

            # for each template interval, calculate the ratio difference
            # rescale the current interval
            for _, bbox in enumerate(bboxes):
                orig_pt0 = max(bbox.orig_rect[0][self.main_axis], self.temp_startline)
                orig_pt1 = min(bbox.orig_rect[1][self.main_axis], self.temp_endline)
                temp_pt0 = max(bbox.temp_rect[0][self.main_axis], self.temp_startline)
                temp_pt1 = min(bbox.temp_rect[1][self.main_axis], self.temp_endline)
                if first_ref <= temp_pt0 <= second_ref:
                    bbox.curr_rect[0] = cal_new_pts_partial(orig_pt0, bbox.orig_rect[0])
                    bbox.curr_proposal_rect[0] = cal_new_pts_partial(temp_pt0, bbox.temp_rect[0])
                if first_ref <= temp_pt1 <= second_ref:
                    bbox.curr_rect[1] = cal_new_pts_partial(orig_pt1, bbox.orig_rect[1])
                    bbox.curr_proposal_rect[1] = cal_new_pts_partial(temp_pt1, bbox.temp_rect[1])

        return bboxes

    def locate_bboxes_in_minor_direction_for_depth(self, bboxes):
        if not bboxes:
            return []

        for _, bbox in enumerate(bboxes):
            bbox.curr_rect3d[0] = cal_new_pts_only_minor(bbox.curr_rect[0], self.main_axis, self.minor_axis_poly_func)
            bbox.curr_rect3d[1] = cal_new_pts_only_minor(bbox.curr_rect[1], self.main_axis, self.minor_axis_poly_func)
            bbox.curr_proposal_rect3d[0] = cal_new_pts_only_minor(
                bbox.curr_proposal_rect[0], self.main_axis, self.minor_axis_poly_func
            )
            bbox.curr_proposal_rect3d[1] = cal_new_pts_only_minor(
                bbox.curr_proposal_rect[1], self.main_axis, self.minor_axis_poly_func
            )

        return bboxes

    @staticmethod
    def locate_bboxes_according2template(bboxes: Union[BBox, List[BBox]], template: Dict):
        """locate items according to chunks template"""
        item_boxes = []
        defect_boxes = []
        if not isinstance(bboxes, list):
            bboxes = [bboxes]
        for box in bboxes:
            if box.is_defect:
                # 如果yolo事先就没有检到chunk，那么chunk内的项点也无法继续检测判断，此时将chunk/box当做项点返回，是否报警再决定
                defect_boxes.append(box)
            else:
                label = box.label
                for item in template[label]:
                    name, num2check = get_label_num2check(item["label"])
                    item_box = BBox(label=item["label"], name=name, num2check=num2check)
                    item_box.orig_rect = item["orig_rect"]
                    item_box.temp_rect = item["temp_rect"]
                    rect = trans_coords_from_chunk2frame(box.curr_rect.to_list(), item["orig_rect"])
                    item_box.curr_proposal_rect = rect
                    item_box.curr_rect = rect
                    item_boxes.append(item_box)
        return item_boxes, defect_boxes

    def _dev_generate_anchors_img_(
            self, anchor_bboxes, save_path, test_img, img_h, img_w, aux="anchors", label_list=[]
        ):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        w_logger.info(">>> Start cropping...")
        for idx, bbox in enumerate(anchor_bboxes):
            if bbox.name in label_list or len(label_list) == 0:
                new_template = deepcopy(LABELME_TEMPLATE)
                new_rectangle = deepcopy(LABELME_RECT_TEMPLATE)

                # get proposal rect in frame points
                curr_proposal_rect = cal_coord_by_ratio_adjustment(
                    bbox.temp_rect,
                    self.temp_startline,
                    self.temp_carspan,
                    self.test_startline,
                    self.test_carspan,
                    self.axis,
                )
                curr_rect = cal_coord_by_ratio_adjustment(
                    bbox.orig_rect,
                    self.temp_startline,
                    self.temp_carspan,
                    self.test_startline,
                    self.test_carspan,
                    self.axis,
                )

                # proposal rect from frame points to local pixel values
                proposal_rect_p = frame2rect(
                    curr_proposal_rect, self.test_startline, img_h, img_w, start_minor_axis_fp=0, axis=self.axis
                )
                curr_rect_p = frame2rect(
                    curr_rect, self.test_startline, img_h, img_w, start_minor_axis_fp=0, axis=self.axis
                )

                # proposal rect image
                proposal_rect_img = crop_segmented_rect(test_img, proposal_rect_p)
                new_rectangle["label"] = bbox.label
                _, _, imageWidth, imageHeight = xyxy2left_xywh(proposal_rect_p)
                new_rectangle_points = xyxy_nested(curr_rect_p, proposal_rect_p)

                # TODO
                if imageHeight * imageWidth < 640 * 640:
                    r = 1
                elif imageHeight * imageWidth < 1280 * 1280:
                    r = 0.5
                elif imageHeight * imageWidth < 2560 * 2560:
                    r = 0.25
                else:
                    r = 0.125
                proposal_rect_img = resize_img(proposal_rect_img, r)
                imageHeight, imageWidth = proposal_rect_img.shape
                new_rectangle_points = [[j[0] * r, j[1] * r] for j in new_rectangle_points]

                new_rectangle["points"] = new_rectangle_points

                fname = concat_str(
                    aux,
                    self.qtrain_info.minor_train_code,
                    self.qtrain_info.train_num,
                    self.qtrain_info.train_sn,
                    self.qtrain_info.channel,
                    self.qtrain_info.carriage,
                    bbox.name,
                    idx,
                )
                new_template["shapes"].append(new_rectangle)
                new_template["imageHeight"] = int(imageHeight)
                new_template["imageWidth"] = int(imageWidth)
                new_template["imagePath"] = fname + ".jpg"
                img_fname = save_path / (fname + ".jpg")
                cv2.imwrite(str(img_fname), proposal_rect_img)
                json_fname = save_path / (fname + ".json")
                write_json(str(json_fname), new_template)
                w_logger.info(f">>> {fname}.")

    def locate_bboxes_using_static_chunks(
            self, bboxes: List[BBox], template: Dict, test_img: np.ndarray, 
            img_h: int, img_w: int, hist_record: Optional[Dict]=None
        ) -> List[Optional[Tuple[str, BBox]]]:
        """locate bboxes according to multi-classes/single-class static template. 1) cls chunk. 2) get the corresponding template and locate items.

        Args:
            bboxes (List[BBox]): including 3 types: [unclassified chunk, normal chunk, the bbox for classification]
            template (Dict): _description_
            test_img (np.ndarray): _description_
            img_h (int): _description_
            img_w (int): _description_
            hist_record (Optional[Dict], optional): _description_. Defaults to None.

        Returns:
            List[Optional[Tuple[str, BBox]]]: _description_
        """
        
        res = []
        if len(bboxes)==0:
            return res

        cls_static_chunks = self.local_params.get('cls_static_chunks', {}) # dict, see channel_params.yaml
        if len(cls_static_chunks)>0:
            for box in bboxes:
                label = box.label
                if label in cls_static_chunks:
                    class_id = None
                    if hist_record is not None and self.qtrain_info.train_num in hist_record:
                        if label in hist_record[self.qtrain_info.train_num]:
                            class_id = hist_record[self.qtrain_info.train_num][label]  # read class_id from hist_record

                    if class_id is None:
                        proposal_rect_p = frame2rect(box.curr_rect, self.test_startline, img_h, img_w, axis=self.axis)
                        img = crop_segmented_rect(test_img, proposal_rect_p)
                        cfg = cls_static_chunks[label]
                        classifier = ClsInfer(cfg.model_type, cfg.model_params, cfg.model_path, self.device, self.mac_password)
                        classifier.get_transform_funs(cfg.transform_funs)
                        class_id = classifier.infer(img)

                    # reset static chunk's label, unclassified static chunk -> classified static chunk.
                    static_chunks = [tmp_box for tmp_box in bboxes if tmp_box.label in cls_static_chunks[label]["cls_items"]]
                    for static_chunk in static_chunks:
                        static_chunk.label = f"{static_chunk.label}{class_id}"
                
                #         # get the corresponding template and items.
                #         item_boxes, _ = self.locate_bboxes_according2template([static_chunk], template)
                #         res.append((static_chunk.label, item_boxes))
                # else:
                #     pass
        for box in bboxes:
            label = box.label
            if label not in cls_static_chunks:
                item_bboxes, _ =  self.locate_bboxes_according2template(box, template)
                res.append( (label, item_bboxes) ) 
        return res
    
    def locate_bboxes_using_dynamic_chunks(self, area_bboxes: List[BBox], template: Dict, test_img: np.ndarray, img_h: int, img_w: int) -> List[Optional[Dict]]:
        """
            locate dynamic items/static_chunks in areas(namely, dynamic chunks). 
            1) infer area_img by yolo. 
            2) If yolo output is a static chunk, the static template is read and the internal items are located.
            3) if yolo output is a item, locate directly.
        """

        res = []
        if len(area_bboxes)==0:
            return res

        area_dict = dict()
        # group by area_label, the models are loaded only once.
        for area in area_bboxes:
            name = area.name
            if name in area_dict:
                area_dict[name].append(area)
            else:
                area_dict[name] = [area]
        
        for area_label, areas in area_dict.items():
            
            cfg = self.local_params["det_dynamic_chunks"][area_label]
            model = YoloInfer(cfg.model_path, self.device, cfg.imgsz, mac_password=self.mac_password)

            for area in areas:
                proposal_rect_p = frame2rect(area.curr_rect, self.test_startline, img_h, img_w, axis=self.axis)
                img = crop_segmented_rect(test_img, proposal_rect_p)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                outputs = model.infer(img, conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres)
                chunk_res = dict() # 一个area可能要检测多个同类static chunk, 用dict存储
                for out in outputs:
                    class_id, xywh = out[0], out[1:5]
                    name, num2check = get_label_num2check(cfg.label_translator[class_id])
                    # new_box is a static chunk if in template else item.
                    new_box = BBox(label=cfg.label_translator[class_id], name=name, num2check=num2check) 
                    new_box.curr_rect = yolo_xywh2xyxy_v2(xywh, area.curr_rect)
                    new_box.curr_proposal_rect = yolo_xywh2xyxy_v2(xywh, area.curr_rect)

                    new_items = self.locate_bboxes_according2template(new_box, template)[0] if new_box.label in template else [new_box]
                    if new_box.label in chunk_res:
                        chunk_res[new_box.label].append( (xywh[:2], new_items) )
                    else:
                        chunk_res[new_box.label] = [ (xywh[:2], new_items) ]

                res.append(chunk_res)
        
        return res
    
    def locate_one_carriage(
            self, anchor_bboxes, item_bboxes, static_chunk_bboxes, dynamic_chunk_bboxes, 
            test_img, test_startline, test_endline, img_h, img_w, chunk_template=None, hist_record=None
        ):
        self.update_test_traininfo(test_startline, test_endline)
        anchor_bboxes = self.locate_anchors_yolo(anchor_bboxes, test_img, img_h, img_w)
        self.get_affine_transformation(anchor_bboxes)
        item_bboxes = self.locate_bboxes_according2anchors(item_bboxes)
        static_chunk_bboxes =  self.locate_bboxes_according2anchors(static_chunk_bboxes)
        dynamic_chunk_bboxes =  self.locate_bboxes_according2anchors(dynamic_chunk_bboxes)

        static_chunk_bboxes_res = self.locate_bboxes_using_static_chunks(static_chunk_bboxes, chunk_template, test_img, img_h, img_w, hist_record)
        dynamic_chunk_bboxes_res = self.locate_bboxes_using_dynamic_chunks(dynamic_chunk_bboxes, chunk_template, test_img, img_h, img_w)
        return anchor_bboxes, item_bboxes, static_chunk_bboxes_res, dynamic_chunk_bboxes_res


    def run(self, anchor_bboxes, item_bboxes, static_chunk_bboxes, dynamic_chunk_bboxes, chunk_template=None, hist_record=None):
        
        # 暂不考虑3d当做一通道单独处理情况
        anchors, normal_items, static_chunks, dynamic_chunks = self.locate_one_carriage(
            deepcopy(anchor_bboxes), deepcopy(item_bboxes), deepcopy(static_chunk_bboxes), deepcopy(dynamic_chunk_bboxes), 
            self.qtrain_info.curr_train2d.img, self.qtrain_info.curr_train2d.startline, self.qtrain_info.curr_train2d.endline,
            self.qtrain_info.curr_train2d.img_h, self.qtrain_info.curr_train2d.img_w, chunk_template, hist_record
        )
        self.qtrain_info.curr_train2d.anchors = anchors 
        self.qtrain_info.curr_train2d.normal_items = normal_items
        self.qtrain_info.curr_train2d.static_chunks = static_chunks
        self.qtrain_info.curr_train2d.dynamic_chunks = dynamic_chunks

        if self.qtrain_info.hist_train2d is not None:
            anchors, normal_items, static_chunks, dynamic_chunks = self.locate_one_carriage(
                deepcopy(anchor_bboxes), deepcopy(item_bboxes), deepcopy(static_chunk_bboxes), deepcopy(dynamic_chunk_bboxes),
                self.qtrain_info.hist_train2d.img, self.qtrain_info.hist_train2d.startline, self.qtrain_info.hist_train2d.endline,
                self.qtrain_info.hist_train2d.img_h, self.qtrain_info.hist_train2d.img_w, chunk_template, hist_record
            )
            self.qtrain_info.hist_train2d.anchors = anchors 
            self.qtrain_info.hist_train2d.normal_items = normal_items
            self.qtrain_info.hist_train2d.static_chunks = static_chunks
            self.qtrain_info.hist_train2d.dynamic_chunks = dynamic_chunks
        
        self.collect_all_items()
    
    def collect_all_items(self,):

        def collect_fun(curr_items: List[BBox], hist_items: Optional[List[BBox]]=None):
            if hist_items is None:
                return curr_items
            res = []
            for curr_item, hist_item in zip(curr_items, hist_items):
                curr_item.hist_rect = hist_item.curr_rect
                curr_item.hist_rect3d = hist_item.curr_rect3d
                curr_item.hist_proposal_rect = hist_item.curr_proposal_rect
                curr_item.hist_proposal_rect3d = hist_item.curr_proposal_rect3d
                res.append(curr_item)
            return res

        all_items = []
        if self.qtrain_info.hist_train2d is not None:
            
            all_items += collect_fun(self.qtrain_info.curr_train2d.normal_items, self.qtrain_info.hist_train2d.normal_items)
                
            for curr_static_chunk, hist_static_chunk in zip(self.qtrain_info.curr_train2d.static_chunks, self.qtrain_info.hist_train2d.static_chunks):
                curr_static_chunk_label, curr_static_chunk_items = curr_static_chunk # curr_static_chunk: tuple
                hist_static_chunk_label, hist_static_chunk_items = hist_static_chunk
                # 这两个不一定相同，如受电弓测试车和历史车状态不同
                if curr_static_chunk_label == hist_static_chunk_label:
                    all_items += collect_fun(curr_static_chunk_items, hist_static_chunk_items)
                else:
                    all_items += collect_fun(curr_static_chunk_items)

            for curr_dynamic_chunk, hist_dynamic_chunk in zip(self.qtrain_info.curr_train2d.dynamic_chunks, self.qtrain_info.hist_train2d.dynamic_chunks):
                # curr_dynamic_chunk = {curr_dynamic_chunk_label: [(static_chunk_centerXY, static_chunk_items), (), ...]}
                # 必须以测试车为主: 受电弓升降不同，其历史图项点没有
                for chunk_label, curr_chunk_info in curr_dynamic_chunk.items(): 
                    if chunk_label in hist_dynamic_chunk:
                        hist_chunk_info = hist_dynamic_chunk[chunk_label]

                        # 匈牙利图匹配算法
                        curr_static_chunk_centers = np.array([i[0] for i in curr_chunk_info])
                        hist_static_chunk_centers = np.array([i[0] for i in hist_chunk_info])
                        match_res = hungary_match(curr_static_chunk_centers, hist_static_chunk_centers)
                        for i in range(len(curr_chunk_info)):
                            if i in match_res:
                                j = match_res[i]
                                all_items += collect_fun(curr_chunk_info[i][1], hist_chunk_info[j][1])
                            else:
                                all_items += collect_fun(curr_chunk_info[i][1])  # 没匹配上，测试车检测到动态item/chunk，历史车没有
                    
                    # 测试车检测到动态item/chunk，历史车没有
                    else:
                        tmp_curr_chunk_items = list(itertools.chain(*[i[1] for i in curr_chunk_info]))
                        all_items += collect_fun(tmp_curr_chunk_items)

        else:
            all_items += collect_fun(self.qtrain_info.curr_train2d.normal_items)
            for curr_static_chunk in self.qtrain_info.curr_train2d.static_chunks:
                curr_static_chunk_label, curr_static_chunk_items = curr_static_chunk
                all_items += collect_fun(curr_static_chunk_items)
            for curr_dynamic_chunk in self.qtrain_info.curr_train2d.dynamic_chunks:
                # curr_dynamic_chunk = {curr_dynamic_chunk_label: [(static_chunk_centerXY, static_chunk_items), (), ...]}
                for chunk_label, curr_chunk_info in curr_dynamic_chunk.items():
                    tmp_curr_chunk_items = list(itertools.chain(*[i[1] for i in curr_chunk_info]))
                    all_items += collect_fun(tmp_curr_chunk_items)

        if self.qtrain_info.curr_train3d is not None or self.qtrain_info.hist_train3d is not None:
            map3d_minor_axis_affine_matrix = self.local_params['map3d_minor_axis_affine_matrix']
            map3d_major_axis_affine_matrix = self.local_params.get('map3d_major_axis_affine_matrix', [1,0])
            map3d_minor_axis_poly_func = np.poly1d(map3d_minor_axis_affine_matrix)
            map3d_major_axis_poly_func = np.poly1d(map3d_major_axis_affine_matrix)
            map3d_for_bboxes(all_items, self.axis, map3d_minor_axis_poly_func, map3d_major_axis_poly_func)

        self.qtrain_info.item_bboxes = all_items
