import os
import zipfile
import numpy as np
import open3d as o3d
from pathlib import Path


class imread3d_decorator:
    def __init__(self, h=None, w=None, inner_ext=".data"):
        self.inner_ext = inner_ext
        if h is not None and w is None:
            l = h
            self.reshape_image = lambda img, img_size: img.reshape((l, img_size // l))
        elif h is None and w is not None:
            l = w
            self.reshape_image = lambda img, img_size: img.reshape((img_size // l, l))
        elif h is not None and w is not None:
            self.reshape_image = lambda img, img_size: img.reshape((h, w))
        else:
            raise ValueError("Plase input .hx either height or width")

    def __call__(self, path):
        path = Path(path)
        with zipfile.ZipFile(str(path)) as z:
            f_name = path.stem + self.inner_ext
            with z.open(f_name, 'r') as f:
                img_byte = f.read()
                f.seek(0, os.SEEK_END)
                img_int = np.frombuffer(img_byte, np.uint16)
                img_size = len(img_int)
                img_int = self.reshape_image(img_int, img_size)
        return img_int


imread3d_h2000 = imread3d_decorator(h=2000)
imread3d_w1000 = imread3d_decorator(w=1000)


def depth2pts(depth_img, cam_matrix, scale_factor=1, reshape=True):
    # camera intrinsics
    # [[ fx, 0, cx],
    #  [ 0, fy, cy],
    #  [ 0,  0,  1]]
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
    assert len(depth_img.shape) == 2

    h, w = depth_img.shape
    u, v = np.mgrid[0:h, 0:w]
    z = depth_img / scale_factor
    x = (v - cx) * z / fx
    y = (u - cy) * z / fy
    if reshape:
        xyz = np.dstack((x, y, z)).reshape(-1, 3)
    else:
        xyz = np.dstack((x, y, z))
    return xyz


def rm_empty_points(pts):
    pts_mask = np.sum(pts, axis=1) != 0
    return pts[pts_mask]


def ndarray2pcd(new_pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_pts)
    return pcd


def remap_3d_to_2d(d_img, boundary_max=None, boundary_min=None, scale=255):
    # assert boundary_min >= 0, "boundary min must be greater than or equal to zero"
    # assert boundary_max > boundary_min, "Boundary max must be greater than or equal to boundary min"
    # when depth value == 0, that measn there is no valid depth value captured
    if boundary_max is None:
        boundary_max = np.max(d_img[d_img != 0])
        boundary_min = np.min(d_img[d_img != 0])
    scale_d_img = scale * (d_img.astype(np.float32) - boundary_min) / (boundary_max - boundary_min)
    scale_d_img[scale_d_img < 0] = 0
    scale_d_img[scale_d_img > 255] = 255
    return scale_d_img.astype(np.uint8)


def depthImg2pcd(img, cam_matrix, depth_scale=1, depth_trunc=1000):
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
    h, w = img.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    img = o3d.geometry.Image(img.astype(np.float32))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        img, intrinsic, depth_scale=depth_scale, depth_trunc=depth_trunc, project_valid_depth_only=True
    )
    return pcd


def rgbdImg2pcd(rgb_img, d_img, cam_matrix, depth_scale=1, depth_trunc=1000):
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
    h, w = d_img.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    rgb_img = o3d.geometry.Image(rgb_img.astype(np.uint8))
    d_img = o3d.geometry.Image(d_img.astype(np.float32))
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_img, d_img, depth_scale=depth_scale, depth_trunc=depth_trunc
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic, project_valid_depth_only=True)
    return pcd
