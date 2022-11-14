import torch
import math
import numpy as np

# IOU
# def cal_iou(bbox, prebox):
#     # bbox, prebox = [x,y,w,h]
#     # bbox,prebox左上角坐标
#     xmin1, ymin1 = int(bbox[0] - bbox[2] / 2.0), int(bbox[1] - bbox[3] / 2.0)
#     xmax1, ymax1 = int(bbox[0] + bbox[2] / 2.0), int(bbox[1] + bbox[3] / 2.0)

#     xmin2, ymin2 = int(prebox[0] - prebox[2] / 2.0), int(prebox[1] - prebox[3] / 2.0)
#     xmax2, ymax2 = int(prebox[0] + prebox[2] / 2.0), int(prebox[1] + prebox[3] / 2.0)
#     # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
#     xx1 = np.max([xmin1, xmin2])
#     yy1 = np.max([ymin1, ymin2])
#     xx2 = np.min([xmax1, xmax2])
#     yy2 = np.min([ymax1, ymax2])
#     # 计算两个矩形框面积
#     bbox_area = (xmax1 - xmin1) * (ymax1 - ymin1)
#     prebox_area = (xmax2 - xmin2) * (ymax2 - ymin2)
#     inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))  # 计算交集面积
#     iou = inter_area / (bbox_area + prebox_area - inter_area + 1e-6)  # 计算交并比
#     return iou
def cal_iou(bboxes1, bboxes2):
    bboxes1 = np.array(bboxes1, dtype=np.float32)
    bboxes2 = np.array(bboxes2, dtype=np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    ious = inter_area / (area1 + area2 - inter_area + 1e-6)

    if exchange:
        ious = ious.T
    return ious


# GIOU
# def cal_giou(bbox, prebox):
#     # bbox, prebox = [x,y,width,height]
#     # bbox,prebox左上角坐标
#     xmin1, ymin1 = int(bbox[0] - bbox[2] / 2.0), int(bbox[1] - bbox[3] / 2.0)
#     xmax1, ymax1 = int(bbox[0] + bbox[2] / 2.0), int(bbox[1] + bbox[3] / 2.0)

#     xmin2, ymin2 = int(prebox[0] - prebox[2] / 2.0), int(prebox[1] - prebox[3] / 2.0)
#     xmax2, ymax2 = int(prebox[0] + prebox[2] / 2.0), int(prebox[1] + prebox[3] / 2.0)
#     # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
#     xx1 = np.max([xmin1, xmin2])
#     yy1 = np.max([ymin1, ymin2])
#     xx2 = np.min([xmax1, xmax2])
#     yy2 = np.min([ymax1, ymax2])
#     # 计算两个矩形框面积
#     bbox_area = (xmax1 - xmin1) * (ymax1 - ymin1)
#     prebox_area = (xmax2 - xmin2) * (ymax2 - ymin2)
#     inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))  # 计算交集面积
#     iou = inter_area / (bbox_area + prebox_area - inter_area + 1e-6)  # 计算交并比
#     # 计算Ac
#     area_C = (max(xmin1, xmax1, xmin2, xmax2) - min(xmin1, xmax1, xmin2, xmax2)) * (
#         max(ymin1, ymax1, ymin2, ymax2) - min(ymin1, ymax1, ymin2, ymax2)
#     )
#     # 计算并集
#     area_U = bbox_area + prebox_area - inter_area

#     giou = iou - (area_C - area_U) / area_C
#     return giou
def cal_giou(bboxes1, bboxes2):
    bboxes1 = np.array(bboxes1, dtype=np.float32)
    bboxes2 = np.array(bboxes2, dtype=np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    gious = torch.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return gious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        gious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1 + area2 - inter_area
    closure = outer_area
    ious = inter_area / (union + 1e-6)
    gious = ious - (closure - union) / closure
    # gious = inter_area / union - (closure - union) / closure
    gious = torch.clamp(gious, min=-1.0, max=1.0)

    if exchange:
        gious = gious.T
    return gious


# DIOU
def cal_diou(bboxes1, bboxes2):
    bboxes1 = np.array(bboxes1, dtype=np.float32)
    bboxes2 = np.array(bboxes2, dtype=np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:  #
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows), dtype=np.float32)
        exchange = True
    # xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return dious


# CIOU
def cal_ciou(bboxes1, bboxes2):
    bboxes1 = np.array(bboxes1, dtype=np.float32)
    bboxes2 = np.array(bboxes2, dtype=np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows), dtype=np.float32)
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi**2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi**2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    if exchange:
        cious = cious.T
    return cious
