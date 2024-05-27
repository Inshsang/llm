import os
import numpy as np
import warnings
import pickle
import json
from tqdm import tqdm
import torch
import re
import open3d as o3d
import multiprocessing
import warnings
import jsonlines
warnings.filterwarnings("ignore")

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def interpolate_points(point_cloud, target_points=8192):
    n_points = len(point_cloud)

    while n_points < target_points:
        # 计算每个点与其最近邻点的距离
        distances = np.sum((point_cloud[:, np.newaxis] - point_cloud) ** 2, axis=-1)
        np.fill_diagonal(distances, np.inf)  # 避免将点与自身匹配

        # 找到每个点的最近邻点索引
        nearest_indices = np.argmin(distances, axis=1)

        # 对每个点进行插值
        interpolated_points = []
        for i, idx in enumerate(nearest_indices):
            interpolated_points.append((point_cloud[i] + point_cloud[idx]) / 2)

        # 添加插值点到原始点云中
        point_cloud = np.concatenate([point_cloud, np.array(interpolated_points)])

        # 更新点的数量
        n_points = len(point_cloud)

    # 如果点的数量超过目标值，随机选择一些点
    # if n_points > target_points:
    #     indices = np.random.choice(n_points, target_points, replace=False)
    #     point_cloud = point_cloud[indices]

    return point_cloud

def normalize_point_cloud(point_cloud):
    """
    将输入的点云numpy数组归一化
    :param point_cloud: 输入的点云numpy数组，形状为(N, 3)
    :return: 归一化后的点云numpy数组
    """
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    scale = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)))
    point_cloud /= scale
    return point_cloud

def cut_point_cloud(point_cloud, bbox_list):
    cut_parts = []
    for bbox in bbox_list:
        # Extracting xyzwlh from bbox
        x, y, z, width, length, height = bbox['BoundingBox']

        # Finding points within the bbox
        mask = np.stack((
            point_cloud[:, 0] >= x - width / 2,
            point_cloud[:, 0] <= x + width / 2,
            point_cloud[:, 1] >= y - length / 2,
            point_cloud[:, 1] <= y + length / 2,
            point_cloud[:, 2] >= z - height / 2,
            point_cloud[:, 2] <= z + height / 2),axis=0
        )
        mask = np.all(mask, axis=0)

        # Applying the mask to extract points
        cut_part = point_cloud[mask]
        # # 可视化点云
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(cut_part)
        # o3d.visualization.draw_geometries([pcd])
        while (len(cut_part)<8192):
            cut_part = interpolate_points(cut_part)
        cut_part = cut_part[np.random.choice(cut_part.shape[0], 8192, replace=False)]
        # cut_part = farthest_point_sample(cut_part,8192)
        cut_parts.append(cut_part)

    return cut_parts

# def cut_point_cloud(point_cloud, bbox_list):
#     cut_parts = []
#     for bbox in bbox_list:
#         # Extracting xyzwlh from bbox
#         x, y, z, width, length, height = bbox['BoundingBox']
#
#         # Finding points within the bbox
#         mask = np.stack((
#             point_cloud[:, 0] >= x - width / 2,
#             point_cloud[:, 0] <= x + width / 2,
#             point_cloud[:, 1] >= y - length / 2,
#             point_cloud[:, 1] <= y + length / 2,
#             point_cloud[:, 2] >= z - height / 2,
#             point_cloud[:, 2] <= z + height / 2),axis=0
#         )
#         mask = np.all(mask, axis=0)
#
#         # Applying the mask to extract points
#         cut_part = point_cloud[mask]
#
#         if len(cut_part)==0:
#             print(bbox,"########################")
#     return cut_parts

class_mapping = {
    "cabinet": 0,
    "bed": 1,
    "chair": 2,
    "sofa": 3,
    "diningtable": 4,
    "doorway": 5,
    "window": 6,
    "shelf": 7,
    "painting": 8,
    "countertop": 9,
    "desk": 10,
    # "curtain": 11,  #
    "fridge": 12,
    # "showercurtrain": 13,  #
    "toilet": 14,
    "sink": 15,
    # "bathtub": 16,  #
    "garbagecan": 17,
}

detection_gt = []
# save_path = '/media/kou/Data1/htc/LAMM/data/cut_scene_test_label.dat'
# save_path = '/media/kou/Data1/htc/LAMM/data/cut_scene_train_label.dat'
# save_path = '/media/kou/Data1/htc/LAMM/data/cut_scene_test.dat'
save_path = '/media/kou/Data1/htc/LAMM/data/cut_scene_train.dat'

with open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json", "r") as G:
    jsonlines_data = jsonlines.Reader(G)
    for lines in jsonlines_data:
        id = next(iter(lines))
        # if int(id) >= 500:
        #     continue
        if int(id) < 500 or int(id) >= 10000:
            continue

        inclass_box = []
        bbox = lines[id]
        for i in bbox:
            if not len(i):
                continue
            if i['name'].lower() in class_mapping.keys():
                inclass_box.append(i)
        detection_gt.append({'id': int(id), 'bbox': inclass_box})

vision_path_list = []
f = open("/media/kou/Data1/htc/LAMM/data/3D_Instruct/meta_file/Detection.json",'r')
for item in json.load(f):
    one_vision_path = item["pcl"][6:-4]
    vision_path_list.append(one_vision_path)


label = np.load('/media/kou/Data1/htc/LAMM/data/cut_scene_train_label.dat',allow_pickle=True)
p0 = 0
pos = []
for p in label:
    pos.append(p0)
    p0 += p

lenth = 0
for l in detection_gt:
    lenth+=len(l['bbox'])

import sharedmem
# 创建共享数组
list_of_points = sharedmem.empty((lenth, 8192, 3), dtype=np.float32)
list_of_labels = sharedmem.empty(len(detection_gt), dtype=np.int32)

def cal(index):
    global list_of_points, list_of_labels, detection_gt, vision_path_list
    bbox = detection_gt[index]['bbox']
    path = '/media/kou/Data3/htc/scene/' +str(detection_gt[index]['id']) + '.ply'
    # path = '/media/kou/Data3/htc/scene/'+vision_path_list[index]+'.ply'
    points = o3d.io.read_point_cloud(path)
    scene = np.asarray(points.points)
    objs = cut_point_cloud(scene,bbox)
    for i,obj in enumerate(objs):
        list_of_points[pos[index]+i] = obj
    print(index, 'over')
    # list_of_labels[index] = len(bbox)


lock = multiprocessing.Lock()

# 创建线程池，限制最多10个线程
from multiprocessing import Pool
max_processes = 20
pool = Pool(processes=max_processes)

num_jobs = len(detection_gt)  # 总共要执行的任务数
# num_jobs = 1  # 总共要执行的任务数
# 启动进程池
pool.map(cal, range(num_jobs))

# 关闭进程池
pool.close()
pool.join()

# 现在list_of_points和list_of_labels已经被修改
# print(list_of_points)
# print(list_of_labels)


# 将共享数组或列表转换为普通的numpy数组或列表
points_array = np.array(list(list_of_points))
# labels_array = np.array(list(list_of_labels))

# with open(save_path, 'wb') as f:
#     pickle.dump(labels_array, f)
with open(save_path, 'wb') as f:
    pickle.dump(points_array, f)
# with open(save_path, 'wb') as f:
#     pickle.dump([points_array, labels_array], f)