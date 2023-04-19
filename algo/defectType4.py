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
from pystar360.utilities.helper3d import depthImg2pcd
# from pystar360.utilities.helper3d import
from pystar360.utilities.logger import w_logger
import open3d as o3d
from typing import Tuple, List, Optional
from pystar360.base.dataStruct import CarriageInfo, BBox
@algoDecorator
class DetectNutLoose3d_v1(algoBaseABC):
    """
    检测螺帽/螺帽松动（凸出来螺帽的状态）
    本流程通过2d流程定位，curr_rect映射到curr_rect3d 直接映射到3点图像，所以不需要yolo，其中图片裁剪框用的是curr_rect3d

    depth_items:
        "label_name":
            module: "pystar360.algo.defectType4"
            func: "DetectNutLoose3d_v1"
            params:
                top_plane_offset: 3 # 螺帽最高平面允许的深度值波动范围
                contour_expand_pixel: 1 # 螺帽边缘扩张
                min_nonzero_ratio: 0.4 # 如果非零数在深度图占了40%以下，说明深度值丢失
                relative_depth_thres: 17 # 螺帽顶和其周边的深度值差。阈值设置，不能超过这个相对高度不然报警
                min_depth_value: 0 # 这个区域不可能出现比这个小的值 default=0
                at_least_r_depth: None # 螺帽最小需要这个相对高度, 参考用的正常高度。 relative_depth_thres > at_least_r_depth >= top_plane_offset
    """

    TOP_PLANE_OFFSET = 3
    CONTOUR_EXPAND_PIXEL = 2
    RELATIVE_DEPTH_THRES = 100
    MIN_NONZERO_RATIO = 0.4
    MIN_DEPTH_VALUE = 0
    AT_LEAST_R_DEPTH = None

    def __call__(self, item_bboxes_list: List[BBox], curr_train3d: CarriageInfo,  **kwargs) -> List[BBox]:
        # if empty, return empty
        if not item_bboxes_list:
            return []

        top_plane_offset = self.item_params.get("top_plane_offset", self.TOP_PLANE_OFFSET)
        contour_expand_pixel = self.item_params.get("contour_expand_pixel", self.CONTOUR_EXPAND_PIXEL)
        relative_depth_thres = self.item_params.get("relative_depth_thres", self.RELATIVE_DEPTH_THRES)
        min_nonzero_ratio = self.item_params.get("min_nonzero_ratio", self.MIN_NONZERO_RATIO)
        min_depth_value = self.item_params.get("min_depth_value", self.MIN_DEPTH_VALUE)
        at_least_r_depth = self.item_params.get("at_least_r_depth", self.AT_LEAST_R_DEPTH)

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
                continue
            
            if box.curr_rect3d.is_none() or box.curr_rect3d == box.curr_proposal_rect3d:
                box.description = f">>> curr_rect3d is None or didnt locate the accurate coords, skip"
                box.is_detected = 1
                box.index = count
                new_item_bboxes_list.append(box)
                count += 1
                continue
            # get item depth image
            rect_f = box.curr_rect3d
            # convert it to local rect point
            rect_p = frame2rect(rect_f, curr_train3d.startline, curr_train3d.img_h, curr_train3d.img_w, axis=self.axis)
            item_img = crop_segmented_rect(curr_train3d.img, rect_p)
            item_img_h, item_img_w = item_img.shape
            # image preprocess
            item_img = cv2.medianBlur(item_img, 3)

            # img_nonzero = item_img[item_img != 0]
            img_nonzero = item_img[item_img > min_depth_value]
            # if the number of 0 is greather than xx% of the image size, it means the loss of depth value. Skip detection"
            if img_nonzero.size / item_img.size < min_nonzero_ratio:
                box.description = f">>> Depth value loss. Skip depth detection"
                box.is_detected = 1
                box.index = count
                new_item_bboxes_list.append(box)
                count += 1
                break

            # 确定螺栓的顶部深度值，在定位后，通常都是最高的数值平面
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
                (item_img > max_val - top_plane_offset)
                & (item_img < max_val + top_plane_offset)
                & (item_img > min_depth_value)
            )
            hat_height = np.median(item_img[hat_mask])

            try:
                # get nut base height
                mask_rect = self.find_boundary_rect(hat_mask)
                ty = max(0, mask_rect[0][1] - contour_expand_pixel)
                by = min(item_img_h - 1, mask_rect[1][1] + contour_expand_pixel)
                lx = max(0, mask_rect[0][0] - contour_expand_pixel)
                rx = min(item_img_w - 1, mask_rect[1][0] + contour_expand_pixel)
                base_img = item_img[ty:by, lx:rx]
                # base_img = item_img[mask_rect[0][1] : mask_rect[1][1], mask_rect[0][0] : mask_rect[1][0]]
                # get base
                if at_least_r_depth is not None:
                    top_plane_offset = at_least_r_depth
                if base_img.size > 1:
                    base_mask = base_img > max_val - top_plane_offset
                    base_mask = (~base_mask) & (base_img > min_depth_value)
                    base_height = np.median(base_img[base_mask])
                    # base_height = np.percentile(base_img[base_mask], 45)
                else:
                    base_mask = item_img > max_val - top_plane_offset
                    base_mask = (~base_mask) & (item_img > min_depth_value)
                    base_height = np.median(item_img[base_mask])
                    # base_height = np.percentile(item_img[base_mask], 45)
            except:
                base_height = 0
            # print(f"base_height {base_height}")

            # get relative depth
            relative_depth = np.abs(hat_height - base_height)
            # print(f"relative_depth {relative_depth}")

            # detect
            if relative_depth_thres < relative_depth < 100:
                box.is_3ddefect = 1
                box.is_defect = 1
            elif relative_depth > 100:
                # if it exceeds 100 something probably goes wrong, force it to be threshold
                relative_depth = relative_depth_thres

            box.is_detected = 1
            box.value_3d = relative_depth
            box.value_3dthres = relative_depth_thres
            box.description = f"{box.index}; Depth threshold: {box.value_3dthres}; Relative depth: {box.value_3d};"
            box.index = count
            new_item_bboxes_list.append(box)
            count += 1

            if self.debug:
                w_logger.info(box.description)

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

def get_fpfh_feat(pcd: o3d.geometry.PointCloud, voxel_size: int) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    radius_normal = voxel_size * 3
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    radius_feature = voxel_size * 6
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, 
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    
    return pcd, pcd_fpfh

def execute_fast_global_registration(
        src_pcd: o3d.geometry.PointCloud, tar_pcd: o3d.geometry.PointCloud, src_fpfh: o3d.pipelines.registration.Feature,
        tar_fpfh: o3d.pipelines.registration.Feature, voxel_size: int
    ) -> o3d.pipelines.registration.RegistrationResult:
    distance_threshold = voxel_size * 0.5

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_pcd, tar_pcd, src_fpfh, tar_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold)
    )
    return result


@algoDecorator
class DetectLoose3dByAlign(algoBaseABC):
    """
    根据点云匹配程度(点云配准后重叠的点云对比例), 判断检测螺帽/螺帽松动
    depth_items:
        "label_name":
            module: "pystar360.algo.defectType4"
            func: "DetectLoose3dByAlign"
            params:
                ratio_thres: 0.08 # 配准重合比例
                dist_range: [2, 10] # 配准后的点云对距离在范围内，认为该点重合
    """
    PCD_NUM = 10000 # process down_sampling if PCD_NUM > 10000; 
    ICP_MAX_CORR_DIST = 1.0 
    CAMERA_MATRIX = [              # Camera calibration matrix
        [2155.9, 0, 1077.9], 
        [0, 1077.9, 797.75], 
        [0, 0, 1]
    ]
    DEPTH_SCALE = 1                # depth2pcd params
    DEPTH_TRUNC = 1000
    VOXEL_SIZE = 0.5
    DIST_RANGE = [2, 10]
    
    def __call__(self, item_bboxes_list: List[BBox], curr_train3d: CarriageInfo, hist_train3d: CarriageInfo, **kwargs) -> List[BBox]:
        # if empty, return empty
        if not item_bboxes_list:
            return []
        
        ratio_thres = self.item_params['ratio_thres']
        min_dist_thres, max_dist_thres = self.item_params.get('dist_range', self.DIST_RANGE)
        icp_max_corr_dist = self.item_params.get('icp_max_corr_dist', self.ICP_MAX_CORR_DIST)
        voxel_size = self.item_params.get('voxel_size', self.VOXEL_SIZE)
        depth_scale = self.item_params.get('depth_scale', self.DEPTH_SCALE)
        depth_trunc = self.item_params.get('depth_trunc', self.DEPTH_TRUNC)
        camera_matrix = np.array(self.item_params.get('camera_matrix', self.CAMERA_MATRIX))

        new_item_bboxes_list = []
        for count, box in enumerate(item_bboxes_list):

            # if defect, then skip
            if box.is_defect:
                box.description = f">>> It has been defected. Skip depth detection"
                box.is_detected = 1
                box.index = count + 1
                new_item_bboxes_list.append(box)
                continue
            
            if box.curr_rect3d.is_none() or box.curr_rect3d == box.curr_proposal_rect3d:
                box.description = f">>> curr_rect3d is None or didnt locate the accurate coords, skip"
                box.is_detected = 1
                box.index = count
                new_item_bboxes_list.append(box)
                count += 1
                continue

            if box.hist_rect3d.is_none() or box.hist_rect3d == box.hist_proposal_rect3d:
                box.description = f">>> hist_rect3d is None or didnt locate the accurate coords, skip"
                box.is_detected = 1
                box.index = count
                new_item_bboxes_list.append(box)
                count += 1
                continue
            # get item depth image: '.data' format
            curr_rect, hist_rect = box.curr_rect3d, box.hist_rect3d
            curr_rect = frame2rect(curr_rect, curr_train3d.startline, curr_train3d.img_h, curr_train3d.img_w, axis=self.axis)
            hist_rect = frame2rect(hist_rect, hist_train3d.startline, hist_train3d.img_h, hist_train3d.img_w, axis=self.axis)
            curr_data, hist_data = crop_segmented_rect(curr_train3d.img, curr_rect), crop_segmented_rect(hist_train3d.img, hist_rect)

            # depth2pointcloud
            curr_pcd = depthImg2pcd(curr_data, camera_matrix, depth_scale, depth_trunc)
            hist_pcd = depthImg2pcd(hist_data, camera_matrix, depth_scale, depth_trunc)

            if len(np.array(curr_pcd.points)) > self.PCD_NUM:
                curr_pcd = curr_pcd.voxel_down_sample(voxel_size)
            if len(np.array(hist_pcd.points)) > self.PCD_NUM:
                hist_pcd = hist_pcd.voxel_down_sample(voxel_size)
            
            curr_pcd, curr_fpfh = get_fpfh_feat(curr_pcd, voxel_size)
            hist_pcd, hist_fpfh = get_fpfh_feat(hist_pcd, voxel_size)

            # coarse and fine registration
            res_coarse = execute_fast_global_registration(curr_pcd, hist_pcd, curr_fpfh, hist_fpfh, voxel_size)
            res_fine = o3d.pipelines.registration.registration_icp(
                curr_pcd, hist_pcd, icp_max_corr_dist, res_coarse.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            curr_pcd.transform(res_fine.transformation)

            curr_pcd_num, hist_pcd_num = len(np.array(curr_pcd.points)), len(np.array(hist_pcd.points))
            if curr_pcd_num > hist_pcd_num:
                dists = hist_pcd.compute_point_cloud_distance(curr_pcd)
            else: 
                dists = curr_pcd.compute_point_cloud_distance(hist_pcd)
            dists = np.array(dists)

            if curr_pcd_num < self.PCD_NUM:
                out_num = ((dists>min_dist_thres)&(dists<max_dist_thres)).sum()
            else:
                out_num = ((dists>min_dist_thres*2)&(dists<max_dist_thres)).sum()
            
            r = out_num / len(dists)
            if r > ratio_thres:
                box.is_3ddefect = 1
                box.is_defect = 1   
                box.description = f">>> 3ddefect! score={r}, thres={ratio_thres}"
            
            box.is_detected = 1
            box.index = count + 1
            new_item_bboxes_list.append(box)
            if self.debug:
                w_logger.info(box.description)

        return new_item_bboxes_list