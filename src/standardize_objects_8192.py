import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

# =========================
# 你给的函数（原样使用）
# =========================
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
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
        distances = np.sum(
            (point_cloud[:, np.newaxis] - point_cloud[np.newaxis, :]) ** 2,
            axis=-1
        )
        np.fill_diagonal(distances, np.inf)

        nearest_indices = np.argmin(distances, axis=1)

        interpolated_points = []
        for i, idx in enumerate(nearest_indices):
            interpolated_points.append(
                (point_cloud[i] + point_cloud[idx]) / 2
            )

        point_cloud = np.concatenate(
            [point_cloud, np.array(interpolated_points)],
            axis=0
        )
        n_points = len(point_cloud)

    return point_cloud


# =========================
# 路径配置
# =========================
SRC_ROOT = "/data/HTC/Data/dataset/object_npy"
DST_ROOT = "/data/HTC/Data/dataset/objects_1024_npy"
os.makedirs(DST_ROOT, exist_ok=True)

NPOINTS = 1024
NUM_WORKERS = 20

files = sorted([f for f in os.listdir(SRC_ROOT) if f.endswith(".npy")])
print(f"[INFO] Found {len(files)} object npy files")


# =========================
# 单文件处理
# =========================
def process_one(fname):
    src = os.path.join(SRC_ROOT, fname)
    dst = os.path.join(DST_ROOT, fname)

    pts = np.load(src).astype(np.float32)[:, :3]

    if pts.shape[0] < NPOINTS:
        pts = interpolate_points(pts, NPOINTS)

    pts = farthest_point_sample(pts, NPOINTS)

    np.save(dst, pts)


# =========================
# 多进程执行
# =========================
print("[INFO] Start standardizing point clouds...")
with Pool(NUM_WORKERS) as pool:
    list(tqdm(pool.imap_unordered(process_one, files), total=len(files)))

print(f"[OK] All objects saved to {DST_ROOT}")
