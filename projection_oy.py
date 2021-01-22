# coding=utf-8

from __future__ import print_function, division, absolute_import

import argparse
import os
import pickle
import open3d as o3d
import matplotlib.pyplot as plt
import numpy.linalg as LA
import yaml
from scipy.interpolate import griddata

import cv2
import numpy as np

SCALE_FACTOR = 2
frame_id = 0

#get path
def load_data_pair(root):
    # 获取点云和图像对
    pc_root = os.path.join(root, 'pcds')
    img_root = os.path.join(root, 'images')
    pc_file_list = list(sorted(os.listdir(pc_root)))
    img_file_list = list(sorted(os.listdir(img_root)))
    assert len(pc_file_list) == len(img_file_list)
    return pc_file_list,img_file_list


def visualize_colored_pointcloud(pc):
    try:
        from mayavi import mlab
    except ImportError:
        print('mayavi not found, skip visualize')
        return
        # plot rgba points
    mlab.figure('pc', bgcolor=(0.05, 0.05, 0.05))
    # 构建lut 将RGB颜色索引到点
    lut_idx = np.arange(len(pc))
    lut = np.column_stack([pc[:, 4:][:, ::-1], np.ones_like(pc[:, 0]) * 255])
    # plot
    p3d = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], lut_idx, mode='point')
    p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut_idx)
    p3d.module_manager.scalar_lut_manager.lut.table = lut
    # mlab.axes()
    mlab.show()


def undistort_projection(points, intrinsic_matrix, extrinsic_matrix):
    points = np.column_stack([points, np.ones_like(points[:, 0])])
    # 外参矩阵
    points = np.matmul(extrinsic_matrix, points.T, )
    # 内参矩阵
    points = np.matmul(intrinsic_matrix, points[:3, :], ).T
    # 深度归一化
    points[:, :2] /= points[:, 2].reshape(-1, 1)
    return points

def back_projection(points, intrinsic_matrix, extrinsic_matrix):
    # 还原深度
    points[:, :2] *= points[:, 2].reshape(-1, 1)

    # 还原相平面相机坐标
    points[:, :3] = np.matmul(LA.inv(intrinsic_matrix), points[:, :3].T).T
    # 还原世界坐标
    # 旋转平移矩阵
    R, T = extrinsic_matrix[:3, :3], extrinsic_matrix[:3, 3]
    points[:, :3] = np.matmul(LA.inv(R), points[:, :3].T - T.reshape(-1, 1)).T
    return points

def img_to_pc(pc, img, extrinsic_matrix, intrinsic_matrix):
    # project pointcloud to image

    projection_points = undistort_projection(pc[:, :3], intrinsic_matrix, extrinsic_matrix)
    projection_points = np.column_stack([np.squeeze(projection_points), pc[:, 3:]])

    # crop
    projection_points = projection_points[np.where(
        (projection_points[:, 0] > 0) &
        (projection_points[:, 0] < img.shape[1]) &
        (projection_points[:, 1] > 0) &
        (projection_points[:, 1] < img.shape[0])
    )]

    # depth map projection
    depth_map = np.zeros_like(img, dtype=np.float32)
    depth_map[np.int_(projection_points[:, 1]), np.int_(projection_points[:, 0]), 0] = projection_points[:, 2]
    depth_map[np.int_(projection_points[:, 1]), np.int_(projection_points[:, 0]), 1] = projection_points[:, 3]

    available_depth_indices = np.where(depth_map[..., 0] > 0)
    projection_points = np.row_stack([available_depth_indices[1], available_depth_indices[0],
                                      depth_map[available_depth_indices][..., 0],
                                      depth_map[available_depth_indices][..., 1]]).T

    # 图像点云深度匹配
    RGB = img[np.int_(projection_points[:, 1]), np.int_(projection_points[:, 0]), :]

    projection_points = np.column_stack([projection_points, RGB])

    # back projection
    pc = back_projection(projection_points, intrinsic_matrix, extrinsic_matrix)

    global frame_id
    # np.save(os.path.join(root, "RGBPoint", '{:06d}.npy'.format(frame_id)), pc)
    print("saving..")
    frame_id += 1
    showpc(pc)
    return pc


def pc_to_img(pc, img, extrinsic_matrix, intrinsic_matrix):
    # 投影验证
    projection_points = undistort_projection(pc[:, :3], intrinsic_matrix, extrinsic_matrix)

    # 裁切到图像平面
    projection_points = np.column_stack([np.squeeze(projection_points), pc[:, 3:]])
    projection_points = projection_points[np.where(
        (projection_points[:, 0] > 0) &
        (projection_points[:, 0] < img.shape[1]) &
        (projection_points[:, 1] > 0) &
        (projection_points[:, 1] < img.shape[0])
    )]
    #projection_points[:,1]=projection_points[:,1]+512
    # scale
    h,w,c=img.shape
    img=img[512:h,0:w]
    img = cv2.resize(img, (int(img.shape[1] / SCALE_FACTOR), int(img.shape[0] / SCALE_FACTOR)))


    projection_points[:, :2] /= SCALE_FACTOR
    board = np.zeros_like(img)
    # 提取边缘
    #edge = np.uint8(np.absolute(cv2.Laplacian(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F)))
    board[...] = img[..., ::1]
    # 反射率可视化
    colors = plt.get_cmap('gist_ncar_r')(projection_points[:, 3] / 255) ** 2
    for idx in range(3):
        board[np.int_(projection_points[:, 1])-int(512/2), np.int_(projection_points[:, 0]), 2 - idx] = colors[:, idx] * 255

    cv2.imshow('Projection', board)
    cv2.waitKey(0)
    cv2.imwrite('./board.png', board)
    return board

def showpc(points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.visualization.draw_geometries([pc])

def visualize_full_pc():
    pcl=os.listdir('./seq07/pcds')
    for l in pcl:
        pc=np.load(os.path.join('./seq07/pcds',l))
        showpc(pc)

def visualize_cropped_pc():
    pcl=os.listdir('./seq07/pcds')
    for l in pcl:
        pc=np.load(os.path.join('./seq07/pcds',l))
        cpc = pc[np.where(
            (pc[:, 0] > 0) &   #x
            (abs(pc[:, 0]) < 200) & # y
            (pc[:, 2] > -2) # z
        )]
        showpc(cpc)

def visualize_pc_cropped_imageview():
    pcl=os.listdir('./seq07/pcds')
    intrinsic_matrix = np.loadtxt(os.path.join('./seq07/param', 'intrinsic'))
    extrinsic_matrix = np.loadtxt(os.path.join('./seq07/param', 'extrinsic'))
    for l in pcl:
        pc=np.load(os.path.join('./seq07/pcds',l))
        pc = pc[np.where(
            (pc[:, 0] > 0) &   #x
            (abs(pc[:, 0]) < 200) & # y
            (pc[:, 2] > -2) # z
        )]
        projection_points = undistort_projection(pc[:, :3], intrinsic_matrix, extrinsic_matrix)
        # 裁切到图像平面
        projection_points = np.column_stack([np.squeeze(projection_points), pc[:, 3:]])
        projection_points = projection_points[np.where(
            (projection_points[:, 0] > 0) &
            (projection_points[:, 0] < 2048) &
            (projection_points[:, 1] > 0) &
            (projection_points[:, 1] < 1536)
        )]
        pc = back_projection(projection_points, intrinsic_matrix, extrinsic_matrix)
        showpc(pc)

def  visualize_pc_in_imageview():
    root = r"./seq07"

    # load validation data pair
    pcl,iml = load_data_pair(root)

    # load intrinsic and extrinsic parameters
    intrinsic_matrix = np.loadtxt(os.path.join(root, 'param', 'intrinsic'))
    distortion = np.loadtxt(os.path.join(root, 'param', 'distortion'))
    extrinsic_matrix = np.loadtxt(os.path.join(root, 'param', 'extrinsic'))
    tvec = extrinsic_matrix[:3, 3].reshape(-1, 1)
    rvec, _ = cv2.Rodrigues(extrinsic_matrix[:3, :3])

    for i in range(0,len(pcl)):
        cpc=np.load(os.path.join(root,'pcds',pcl[i]))
        cimg=cv2.imread(os.path.join(root,'images',iml[i]))
        cimg=cv2.undistort(cimg,intrinsic_matrix, distortion)
        cpc = cpc[np.where(
            (cpc[:, 0] > 0) &   #x
            (abs(cpc[:, 0]) < 200) & # y
            (cpc[:, 2] > -3) # z
        )]
        pc_to_img(cpc, cimg, extrinsic_matrix, intrinsic_matrix)#将点云投影到图像平面

def visualize_pc_in_pcview():
    root = r"./seq07"
    # load validation data pair
    pcl,iml = load_data_pair(root)

    # load intrinsic and extrinsic parameters
    intrinsic_matrix = np.loadtxt(os.path.join(root, 'param', 'intrinsic'))
    distortion = np.loadtxt(os.path.join(root, 'param', 'distortion'))
    extrinsic_matrix = np.loadtxt(os.path.join(root, 'param', 'extrinsic'))
    tvec = extrinsic_matrix[:3, 3].reshape(-1, 1)
    rvec, _ = cv2.Rodrigues(extrinsic_matrix[:3, :3])

    for i in range(0,len(pcl)):
        cpc=np.load(os.path.join(root,'pcds',pcl[i]))
        cimg=cv2.imread(os.path.join(root,'images',iml[i]))
        cimg=cv2.undistort(cimg,intrinsic_matrix, distortion)
        cpc = cpc[np.where(
            (cpc[:, 0] > 0) &   #x
            (abs(cpc[:, 0]) < 200) & # y
            (cpc[:, 2] > -3) # z
        )]
        img_to_pc(cpc, cimg, extrinsic_matrix, intrinsic_matrix)

def view_colored_pc():
    list=os.listdir("./seq07/RGBPoint")
    for l in list:
        points=np.load(os.path.join("./seq07/RGBPoint",l))
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[:, :3])
        pc.colors=o3d.utility.Vector3dVector(points[:, 4:7]/255.0)
        o3d.visualization.draw_geometries([pc])


if __name__ == '__main__':
    #visualize_full_pc()
    #visualize_cropped_pc()
    visualize_pc_in_imageview()
    #visualize_pc_in_pcview()
    #view_colored_pc()