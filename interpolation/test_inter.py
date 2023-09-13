import open3d as o3d
import numpy as np

# Load the .xyz file
pcd = o3d.io.read_point_cloud("../data/longdress/input_2X/input_ld/longdress_vox10_1051.xyz", format='xyz')

# Build a KDTree
kdtree = o3d.geometry.KDTreeFlann(pcd)

# For each point in the point cloud, find its k-nearest neighbors
k = 8  # Change this value based on your needs
for i in range(np.asarray(pcd.points).shape[0]):
    print(f"Point index: {i}")
    _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], k)
    print(f"Indices of its {k} nearest neighbors: {idx}")
    downpcd_farthest = pcd.farthest_point_down_sample(5000)
    break