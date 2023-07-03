import os
from glob import glob
import open3d as o3d
import numpy as np
import argparse
import torch
from tqdm import tqdm
from einops import rearrange
from models.utils import normalize_point_cloud, add_noise
import h5py


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PU-GAN Test Data Generation Arguments')
    parser.add_argument('--input_pts_num', default=4096, type=int, help='the input points number')
    parser.add_argument('--gt_pts_num', default=8192, type=int, help='the gt points number')
    parser.add_argument('--noise_level', default=0, type=float, help='the noise level')
    parser.add_argument('--jitter_max', default=0.03, type=float, help="jitter max")
    parser.add_argument('--mesh_dir', default='./data/PU-GAN/test/', type=str, help='input mesh dir')
    parser.add_argument('--save_dir', default='./data/PU-GAN/test_training/', type=str, help='output point cloud dir')
    # parser.add_argument('--mesh_dir', default='/home/v-chendwang/test-playground/PU-GCN/data/PU1K/test/original_meshes/', type=str, help='input mesh dir')
    # parser.add_argument('--save_dir', default='/home/v-chendwang/test-playground/PU-GCN/data/PU1K/train/', type=str, help='output point cloud dir')
    parser.add_argument('--patch_size', default=4, type=int, help='number of points per patch')
    args = parser.parse_args()
    
    
    mesh_path = glob(os.path.join(args.mesh_dir, '*.off'))
    training_set_gt = []
    training_set_input = []
    for i, path in tqdm(enumerate(mesh_path), desc='Processing'):
        pcd_name = path.split('/')[-1].replace(".off", ".xyz")
        mesh = o3d.io.read_triangle_mesh(path)
        # input pcd
        # input_pcd = mesh.sample_points_poisson_disk(args.input_pts_num)
        # input_pts = np.array(input_pcd.points)
        
        # gt pcd
        gt_pcd = mesh.sample_points_poisson_disk(args.gt_pts_num)
        gt_kd = o3d.geometry.KDTreeFlann(gt_pcd)
        gt_pts = np.array(gt_pcd.points)
        seed = mesh.sample_points_poisson_disk(int(args.gt_pts_num * 2 / args.patch_size))
        seed_pts = np.array(seed.points)
        
        for j in range(seed_pts.shape[0]):
            [k, idx, _] = gt_kd.search_knn_vector_3d(seed_pts[j], args.patch_size * 2)
            gt_tmp = gt_pts[idx]
            training_set_gt.append(gt_pts[idx].tolist())
            low_res = gt_tmp[np.random.choice(len(gt_tmp), size=args.patch_size, replace=False)]
            
            training_set_input.append(low_res)
        
        
    tmp = np.asarray(training_set_gt)
    tmp2 = np.asarray(training_set_input)
    save_path = os.path.join(args.save_dir, "pugan_poisson_4_poisson_8.h5")
    
    hf = h5py.File(save_path, 'w')
    
    hf.create_dataset('poisson_8', data=tmp)
    hf.create_dataset('poisson_4', data=tmp2)
    hf.close()
    