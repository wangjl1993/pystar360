#!/usr/bin/env python
# -*- coding: utf-8 -*-

# start code
import cv2
import numpy as np

# def imread_decorator(resize_ratio):
#     """Image read (GRAY)"""
#     def func(path, size=None, mode=cv2.IMREAD_GRAYSCALE, r=resize_ratio):
#         img = cv2.imread(path, mode)
#         if size is None:
#             img = cv2.resize(img, None, fx=r, fy=r)
#         else:
#             img = cv2.resize(img, size)
#         return img

#     func.resize_ratio = resize_ratio
#     return func


class imread_decorator:
    def __init__(self, resize_ratio):
        self.resize_ratio = resize_ratio

    def __call__(self, path, size=None, mode=cv2.IMREAD_GRAYSCALE):
        img = cv2.imread(path, mode)
        if size is None:
            img = cv2.resize(img,
                             None,
                             fx=self.resize_ratio,
                             fy=self.resize_ratio)
        else:
            img = cv2.resize(img, size)
        return img


imread_tenth = imread_decorator(0.1)
imread_octa = imread_decorator(0.125)
imread_quarter = imread_decorator(0.25)
imread_half = imread_decorator(0.5)
imread_full = imread_decorator(1)


def resize_img(img, resize_ratio):
    """Resize image"""
    return cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)


def concat_str(*args, separator="_"):
    """Concatenate string"""
    args = list(map(str, args))
    return separator.join(args)


def frame2index(frame, length):
    """Frame to index and pixel shift"""
    shift = int(frame)
    pixel = int((frame % 1) * length)
    return shift, pixel


def index2frame(shift, pixel, length):
    """Image index and pixel shift to Frame name"""
    return shift + pixel / length


def cutpoints2frame(l, cutpoints, length):
    """filename, pixel to frame point"""
    shift = l.index(cutpoints[0])
    return index2frame(shift, cutpoints[-1], length)


def frame2cutpoints(l, frame, length):
    """frame point to filename, pixel"""
    shift, pixel = frame2index(frame, length)
    return (l[shift], pixel)


def read_segmented_img(l, startline, endline, imread, imgsz=None, axis=1):
    """Read continuous image frame and concatenate into one image"""
    assert startline <= endline, "startline <= endline"
    if imgsz:
        img_h, img_w = imgsz
    else:
        img_h, img_w = imread(l[0]).shape

    if axis == 0:
        start_idx, start_p = frame2index(startline, img_h)
        end_idx, end_p = frame2index(endline, img_h)
    else:
        start_idx, start_p = frame2index(startline, img_w)
        end_idx, end_p = frame2index(endline, img_w)

    start_f = l[start_idx]
    end_f = l[end_idx]

    if axis == 0:
        img = imread(start_f)[:, start_p:]
    else:
        img = imread(start_f)[start_p:, :]

    if start_idx != end_idx:
        tmp_l = l[start_idx + 1:end_idx]

        for f in tmp_l:
            tmp_img = imread(f)
            if axis == 0:
                if tmp_img.shape[0] < img_h:
                    addition = np.zeros((img_h - tmp_img.shape[0], img_w))
                    tmp_img = np.concatenate((tmp_img, addition), axis=axis)
            else:
                if tmp_img.shape[1] < img_w:
                    addition = np.zeros((img_h, img_w - tmp_img.shape[1]))
                    tmp_img = np.concatenate((tmp_img, addition), axis=1)

            img = np.concatenate((img, tmp_img), axis=axis)

        tmp_img = imread(end_f)
        img = np.concatenate((img, tmp_img[:end_p, :]), axis=axis) if axis == 0 \
        else np.concatenate((img, tmp_img[:, :end_p]), axis=axis)

    else:
        tmp_img = imread(end_f)
        img = tmp_img[
            start_p:end_p, :] if axis == 0 else tmp_img[:, start_p:end_p]

    return img


def xyxy2xywh(points):
    """left-top right-bottom rect points to center xy width height rect points"""
    h = points[1][1] - points[0][1]
    w = points[1][0] - points[0][0]
    x = 0.5 * (points[1][0] + points[0][0])
    y = 0.5 * (points[1][1] + points[0][1])
    return x, y, w, h


def xyxy2left_xywh(points):
    """left-top right-bottom rect points to left-top width height rect points"""
    h = int(points[1][1] - points[0][1])
    w = int(points[1][0] - points[0][0])
    return points[0][0], points[0][1], w, h


def x2pixel(x, sx, img_w):
    """frame values (float) to pixel (int) in horizontal direction"""
    o = int(np.ceil((x - sx) * img_w))
    o = max(o, 0)
    return o


def y2pixel(y, sy, img_h):
    """frame values (float) to pixel (int) in vertical direction"""
    o = int(np.ceil((y - sy) * img_h))
    o = min(max(o, 0), img_h - 1)
    return o


def frame2rect(points, sx, img_h, img_w, sy=0):
    """frame values (float) to pixel (int) in both horizontal and vertical direction"""
    x_left = x2pixel(points[0][0], sx, img_w)
    y_top = y2pixel(points[0][1], sy, img_h)

    x_right = x2pixel(points[1][0], sx, img_w)
    y_bottom = y2pixel(points[1][1], sy, img_h)
    return [(x_left, y_top), (x_right, y_bottom)]


def pixel2x(x, sx, img_w):
    """pixel value (int) to frame values (float) in horizontal direction"""
    return x / img_w + sx


def pixel2y(y, sy, img_h):
    """pixel value (int) to frame values (float) in vertical direction"""
    return y / img_h + sy


def rect2frame(points, sx, img_h, img_w, sy=0):
    """left-top right bottom rect points to frames value (float)"""
    x_left = pixel2x(points[0][0], sx, img_w)
    y_top = pixel2y(points[0][1], sy, img_h)

    x_right = pixel2x(points[1][0], sx, img_w)
    y_bottom = pixel2y(points[1][1], sy, img_h)

    return [(x_left, y_top), (x_right, y_bottom)]


def get_img_size(f, img_read):
    """return image size"""
    return img_read(f).shape


def get_img_size2(h, w, resize_ratio):
    """return image size according to a given ratio"""
    h = int(h * resize_ratio)
    w = int(w * resize_ratio)
    return h, w


def get_rect_area(points):
    """calculate area in a given rectangle"""
    w = abs(points[1][0] - points[0][0])
    h = abs(points[1][1] - points[0][1])
    return max(w * h, 0)


def imread_acc2_area(points, th1=0.003, th2=0.015):
    area = get_rect_area(points)
    if area < th1:
        imread = imread_full
    elif th1 <= area < th2:
        imread = imread_half
    else:
        imread = imread_quarter
    return imread


def read_segmented_rect(l, rect, imread):
    """Read rectangle"""
    img_h = imread(l[0]).shape[0]
    img = read_segmented_img(l, rect[0][0], rect[1][0], imread)
    y_top, y_bottom = y2pixel(rect[0][1], 0,
                              img_h), y2pixel(rect[1][1], 0, img_h)
    img = img[y_top:y_bottom, :]
    return img


def crop_segmented_rect(img, rect):
    """crop partial region from a 2D image"""
    img = img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
    return img


# def binarize_hist(img, axis=0, thres="mid"):
#     """binarize a 1D histgram"""
#     hist = img.mean(axis=axis)
#     mid, low, high = ostu(hist, return_hl=True)
#     if thres == "mid":
#         th = mid
#     elif thres == "low":
#         th = low
#     elif thres == "high":
#         th = high
#     else:
#         raise ValueError("Use mid, low or high.")
#     tmp = np.zeros_like(hist)
#     tmp[hist <= th] = 0
#     tmp[hist > th] = 1
#     return tmp

# def calc_bin_hist_segment(hist, low_seg=True):
#     """"""
#     if low_seg:
#         hist = np.max(hist) - hist

#     seg_idxes = []
#     start_idx = 0
#     start_seg = False
#     for i in range(len(hist) - 1):
#         if hist[i] == 0 and hist[i + 1] == 1:
#             start_idx = i
#             start_seg = True
#         elif hist[i] == 1 and hist[i + 1] == 0:
#             if start_seg:
#                 end_idx = i + 1
#                 seg_idxes.append((start_idx, end_idx))
#                 start_seg = False

#     return seg_idxes

# def to_template_points(rects):
#     """Change to template format points"""
#     output = []
#     if not rects:
#         return []

#     for r in rects:
#         tmp = {}
#         tmp["label"] = r["label"]
#         tmp["number"] = r["number"]
#         tmp["is_error"] = r["is_error"]
#         tmp["pt0"] = r["cur_points"][0]
#         tmp["pt1"] = r["cur_points"][1]
#         output.append(tmp)
#     return output