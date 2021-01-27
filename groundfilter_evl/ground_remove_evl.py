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
import time
import cv2
import numpy as np
from calibration_kitti import Calibration
SCALE_FACTOR = 1
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

    projection_points[:,1]=projection_points[:,1]+512
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
    #cv2.imwrite('./board.png', board)
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

def o3d_ground_filter(pcl,n=30):
    #@内点的最大距离，@每轮迭代的初始点 @迭代轮次
    #return float64[4,1], list[int]
    plane_model, inliers = pcl.segment_plane(distance_threshold=0.2, ransac_n=n, num_iterations=200)
    #[a, b, c, d] = plane_model
    #print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    plane = pcl.select_by_index(inliers)
    #plane.paint_uniform_color([1.0, 0, 0])#red
    outlier = pcl.select_by_index(inliers, invert=True)# invert selection
    #outlier.paint_uniform_color([0, 0, 0])#black
    #print('ground:',len(inliers))
    #plane.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #o3d.visualization.draw_geometries([plane],point_show_normal=True)
    return plane, outlier

#几何过滤+剔除nan值
def boundary_filter(cpc):
    cpc = cpc[np.where(
        (abs(cpc[:, 0]) < 200) &   #x
        (abs(cpc[:, 0]) < 200) & # y
        (abs(cpc[:, 2]) < 10) # z
    )]
    return cpc

def crop_pc(pc, img, extrinsic_matrix, intrinsic_matrix):
    pc=np.asarray(pc.points)
    #a=np.zeros((len(pc[:,0]),1))
    #b=pc
    pc=np.hstack((pc,np.ones((len(pc[:,0]),1))))
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
    pcs = o3d.geometry.PointCloud()
    pcs.points = o3d.utility.Vector3dVector(pc[:, :3])
    return pcs

def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=512, height=512)
    opt = vis.get_render_option()
    opt.point_size = 1
    for pc in pcd:
        vis.add_geometry(pc)
    vis.run()
    time.sleep(0.5)
    vis.destroy_window()

def rs_eval():
    pcl=os.listdir('./seq07/raw')
    intrinsic_matrix = np.loadtxt(os.path.join('./seq07/param', 'intrinsic'))
    distortion = np.loadtxt(os.path.join('./seq07/param', 'distortion'))
    extrinsic_matrix = np.loadtxt(os.path.join('./seq07/param', 'extrinsic'))
    tvec = extrinsic_matrix[:3, 3].reshape(-1, 1)
    rvec, _ = cv2.Rodrigues(extrinsic_matrix[:3, :3])
    image=np.zeros([1536,2048,3])
    info=[]

    for l in pcl:
        pc=np.load(os.path.join('./seq07/raw',l))
        pc=boundary_filter(pc)
        full_frame=pc.shape[0]
        pcs = o3d.geometry.PointCloud()
        pcs.points = o3d.utility.Vector3dVector(pc[:, :3])
        fov_wg=crop_pc(pcs,image,extrinsic_matrix,intrinsic_matrix)
        #o3d.visualization.draw_geometries([pcs])
        plane, outlier=o3d_ground_filter(pcs,10)
        fov_wog=crop_pc(outlier,image,extrinsic_matrix,intrinsic_matrix)
        #re painting
        plane.paint_uniform_color([1.0, 0, 0])#red
        outlier.paint_uniform_color([0, 0, 0])#black
        fov_wog.paint_uniform_color([0,1,0]) #green
        fov_wg.paint_uniform_color([0,0,1]) #blue
        #
        full_non_ground=np.asarray(outlier.points).shape[0]
        fov_wg_count=np.asarray(fov_wg.points).shape[0]
        fov_wog_count=np.asarray(fov_wog.points).shape[0]
        #print(full_non_ground/full_frame,fov_wg_count/full_frame,fov_wog_count/full_frame)
        info.append([full_non_ground/full_frame,fov_wg_count/full_frame,fov_wog_count/full_frame])
        #custom_draw_geometry([plane,outlier,fov_wg,fov_wog])

    np.savetxt('rs-ruby.txt',info,fmt='%0.3f')
    info=np.asarray(info)
    print(np.mean(info[:,0]),np.mean(info[:,1]),np.mean(info[:,2]))

def read_bin(p):
    return np.fromfile(p, dtype=np.float32).reshape(-1, 4)

def load_kitti_calib(calib_file):
    """
    load projection matrix
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 5)

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

def Kitti_pc_to_img(pc, img, extrinsic_matrix, intrinsic_matrix):
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

    #img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))
    board = np.zeros_like(img)
    # 提取边缘
    #edge = np.uint8(np.absolute(cv2.Laplacian(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F)))
    #board[...] = img[..., ::1]
    # 反射率可视化
    colors = plt.get_cmap('gist_ncar_r')(projection_points[:, 3] / 255) ** 2
    for idx in range(3):
        board[np.int_(projection_points[:, 1]), np.int_(projection_points[:, 0]), 2-idx] = colors[:,idx]*255
    cv2.imshow('Projection', board)
    cv2.waitKey(0)
    return board

def kitti_eval():
    path=r'D:\ov\sequences'
    count=0
    image=np.zeros([376,1241,3])
    info=[]
    for i in range(0,11):#
        seq=str(i).zfill(2)
        pc_path=os.listdir(os.path.join(path,seq,'velodyne'))
        calib=load_kitti_calib(os.path.join(path,seq,'calib.txt'))#Calibration(os.path.join(path,seq,'calib.txt'))#load_kitti_calib(os.path.join(path,seq,'calib.txt'))
        intrinsic_matrix=calib['P2'][:,:3]
        extrinsic_matrix=np.vstack((calib['Tr_velo2cam'],[0,0,0,1]))
        #vo dataset contains different images
        if seq=='03':
            img_shape=[375,1242,3]#img.shape
        elif seq in ['00','01','02']:
            img_shape=[376,1241,3]#img.shape
        else:
            img_shape=[370,1226,3]#img.shape
        img=np.ones(img_shape)
        for pcf in range(0,len(pc_path),100):
            count+=1
            pc=read_bin(os.path.join(path,seq,'velodyne',pc_path[pcf]))
            pc=boundary_filter(pc)

            full_frame=pc.shape[0]#整帧点云
            #to o3d
            pcs = o3d.geometry.PointCloud()
            pcs.points = o3d.utility.Vector3dVector(pc[:, :3])
            #对整帧点云进行相机fov裁剪
            fov_wg=crop_pc(pcs,image,extrinsic_matrix,intrinsic_matrix)
            #整帧地面过滤
            plane, outlier=o3d_ground_filter(pcs,10)
            #对非地面点云进行相机FOV裁剪
            fov_wog=crop_pc(outlier,image,extrinsic_matrix,intrinsic_matrix)
            #re painting
            plane.paint_uniform_color([1.0, 0, 0])#red
            outlier.paint_uniform_color([0, 0, 0])#black
            fov_wog.paint_uniform_color([0,1,0]) #green
            fov_wg.paint_uniform_color([0,0,1]) #blue
            #统计剩余点云
            full_non_ground=np.asarray(outlier.points).shape[0]#整帧非地面
            fov_wg_count=np.asarray(fov_wg.points).shape[0]#相机FOV
            fov_wog_count=np.asarray(fov_wog.points).shape[0]#相机FOV非地面
            print(full_non_ground/full_frame,fov_wg_count/full_frame,fov_wog_count/full_frame)
            info.append([full_non_ground/full_frame,fov_wg_count/full_frame,fov_wog_count/full_frame])
            #custom_draw_geometry([fov_wog,fov_wg])
            #custom_draw_geometry([plane,outlier,fov_wog,fov_wg])
    np.savetxt('kitti.txt',info,fmt='%0.3f')
    info=np.asarray(info)
    print(np.mean(info[:,0]),np.mean(info[:,1]),np.mean(info[:,2]))

if __name__ == '__main__':
    kitti_eval()
    #rs_eval()