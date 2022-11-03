################################################################################
#### 松动（3D）类故障算法汇总
################################################################################

import cv2
import copy
import numpy as np
from sympy import O

from pystar360.algo.algoBase import algoBaseABC, algoDecorator
from pystar360.yolo.inference import YoloInfer, yolo_xywh2xyxy_v2
from pystar360.utilities.helper import crop_segmented_rect, frame2rect

# from pystar360.utilities.helper3d import
from pystar360.utilities._logger import d_logger


@algoDecorator
class DetectNutLoose3d_v1(algoBaseABC):
    """
    检测螺帽/螺帽松动（凸出来螺帽状态）
    本流程通过2d流程定位，curr_rect映射到curr_rect3d 直接映射到3点图像，所以不需要yolo，其中图片裁剪框用的是curr_rect3d

    depth_items:
        "label_name":
            module: "pystar360.algo.defectType4"
            func: "DetectNutLoose3d_v1"
            params:
                top_plane_offset: 3 # 螺帽最高平面允许的深度值波动范围
                contour_expand_pixel: 1 # 螺帽边缘扩张
                min_nonzero_ratio: 0.4 # 如果非零数在深度图占了40%以下，说明深度值丢失
                depth_thres: 17 # 螺帽顶和其周边的深度值差。阈值设置
    """

    TOP_PLANE_OFFSET = 3
    CONTOUR_EXPAND_PIXEL = 2
    DEPTH_THRES = 100
    MIN_NONZERO_RATIO = 0.4

    def __call__(self, item_bboxes_list, test_img, test_startline, img_h, img_w, **kwargs):
        # if empty, return empty
        if not item_bboxes_list:
            return []

        top_plane_offset = self.item_params.get("top_plane_offset", self.TOP_PLANE_OFFSET)
        contour_expand_pixel = self.item_params.get("contour_expand_pixel", self.CONTOUR_EXPAND_PIXEL)
        depth_thres = self.item_params.get("depth_thres", self.DEPTH_THRES)
        min_nonzero_ratio = self.item_params.get("min_nonzero_ratio", self.MIN_NONZERO_RATIO)

        # iterate
        new_item_bboxes_list = []
        count = 1
        for _, box in enumerate(item_bboxes_list):

            # if defect, then skip
            if box.is_defect:
                box.description = f">>> It has been defected. Skip depth detection"
                box.is_detected = 1
                box.index = count
                new_item_bboxes_list.append(box)
                count += 1
                break

            # get item depth image
            rect_f = box.curr_rect3d
            # convert it to local rect point
            rect_p = frame2rect(rect_f, test_startline, img_h, img_w, axis=self.axis)
            item_img = crop_segmented_rect(test_img, rect_p)
            item_img_h, item_img_w = item_img.shape
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            img_nonzero = item_img[item_img != 0]
            # if the number of 0 is greather than 70% of the image size, it means the loss of depth value. Skip detection"
            if img_nonzero.size / item_img.size < min_nonzero_ratio:
                box.description = f">>> Depth value loss. Skip depth detection"
                box.is_detected = 1
                box.index = count
                new_item_bboxes_list.append(box)
                count += 1
                break

            # 螺栓的顶部，在定位后，通常都是最高的数值
            hist, bin_edges = np.histogram(
                img_nonzero.flatten(), bins=list(range(img_nonzero.min(), img_nonzero.max() + 1))
            )
            hist, bin_edges = np.append(hist[::-1], [0]), bin_edges[::-1]

            find_hat_plane = 0
            for idx, (c, d) in enumerate(zip(hist, bin_edges)):
                if idx > 4:
                    break
                if c > 9:
                    find_hat_plane = 1
                    max_val = d
                    break

            if not find_hat_plane:
                max_val = np.max(img_nonzero)

            # get hat height
            hat_mask = (
                (item_img > max_val - top_plane_offset) & (item_img < max_val + top_plane_offset) & (item_img != 0)
            )
            hat_height = np.median(item_img[hat_mask])
            # print("****", max_val)
            # print(item_img[hat_mask])
            # print(f"hat_height {hat_height}")

            # get nut base height
            mask_rect = self.find_boundary_rect(hat_mask)
            ty = max(0, mask_rect[0][1] - contour_expand_pixel)
            by = min(item_img_h - 1, mask_rect[1][1] + contour_expand_pixel)
            lx = max(0, mask_rect[0][0] - contour_expand_pixel)
            rx = min(item_img_w - 1, mask_rect[1][0] + contour_expand_pixel)
            base_img = item_img[ty:by, lx:rx]
            # base_img = item_img[mask_rect[0][1] : mask_rect[1][1], mask_rect[0][0] : mask_rect[1][0]]
            # get base
            if base_img.size > 1:
                base_mask = base_img > max_val - top_plane_offset
                base_mask = (~base_mask) & (base_img != 0)
                # base_height = np.median(base_img[base_mask])
                base_height = np.percentile(base_img[base_mask], 45)
            else:
                base_mask = item_img > max_val - top_plane_offset
                base_mask = (~base_mask) & (item_img != 0)
                # base_height = np.median(item_img[base_mask])
                base_height = np.percentile(item_img[base_mask], 45)
            # print(f"base_height {base_height}")

            # get relative depth
            relative_depth = np.abs(hat_height - base_height)
            # print(f"relative_depth {relative_depth}")
            if relative_depth > depth_thres:
                box.is_3ddefect = 1
                box.is_defect = 1

            box.is_detected = 1
            box.value_3d = relative_depth
            box.value_3dthres = depth_thres
            box.description = f"Depth threshold: {box.value_3dthres}: Relative depth: {box.value_3d};"
            box.index = count
            new_item_bboxes_list.append(box)
            count += 1

        return new_item_bboxes_list

    def find_boundary_rect(self, mask):

        l_x, t_y, r_x, b_y = 0, 0, 0, 0
        h, w = mask.shape

        row_sum = mask.sum(axis=0)
        for i in range(w):
            if row_sum[i] != 0:
                l_x = i
                break

        for i in range(max(w - 1, l_x), l_x, -1):
            if row_sum[i] != 0:
                r_x = i
                break

        l_x, r_x = min(l_x, r_x), max(l_x, r_x)

        col_sum = mask.sum(axis=1)
        for i in range(h):
            if col_sum[i] != 0:
                t_y = i
                break

        for i in range(max(h - 1, t_y), t_y, -1):
            if col_sum[i] != 0:
                b_y = i
                break

        t_y, b_y = min(t_y, b_y), max(t_y, b_y)

        return [[l_x, t_y], [r_x, b_y]]


@algoDecorator
class DetectBoltLoose3d(algoBaseABC):
    """检测--螺栓/螺杆--松动"""

    def __call__(self):
        pass
