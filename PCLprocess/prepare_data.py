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
    parser = argparse.ArgumentParser(description='8i Train Data Generation Arguments')
    parser.add_argument('--up_rate', default=2, type=int, help='the upsampling ratio')
    parser.add_argument('--patch_size', default=8, type=int, help='the number of points per patch')
    parser.add_argument('--raw_dir', default='/home/chendong/PU-LUT/data/longdress/Ply/', type=str, help='input mesh dir')
    parser.add_argument('--save_dir', default='/home/chendong/PU-LUT/data/longdress/', type=str, help='output point cloud dir')
    args = parser.parse_args()

    dir_name = 'input_' + str(int(args.up_rate)) + 'X'
    
    input_save_dir = os.path.join(args.save_dir, dir_name, 'input_ld')
    if not os.path.exists(input_save_dir):
        os.makedirs(input_save_dir)
    gt_save_dir = os.path.join(args.save_dir, dir_name, 'gt_ld')
    if not os.path.exists(gt_save_dir):
        os.makedirs(gt_save_dir)
        
    raw_path = glob(os.path.join(args.raw_dir, '*.ply'))
    training_set_gt = []
    training_set_input = []
    for i, path in tqdm(enumerate(raw_path), desc='Processing'):
        pcd = o3d.io.read_point_cloud(path)
        pcd_name = path.split('/')[-1].replace(".ply", ".xyz")
        
        gt_pts = np.asarray(pcd.points)
        # input_pts = np.asarray(pcd.uniform_down_sample(2).points)
        
        
        # gt_save_path = os.path.join(gt_save_dir, pcd_name) 
        # input_save_path = os.path.join(input_save_dir, pcd_name)
        # np.savetxt(input_save_path, input_pts, fmt='%.6f')
        # np.savetxt(gt_save_path, gt_pts, fmt='%.6f')
        
        gt_kd = o3d.geometry.KDTreeFlann(pcd)
        seed_pts = np.asarray(pcd.uniform_down_sample(8).points)
        # seed_pts = np.array(seed.points)
        
        for j in range(seed_pts.shape[0]):
            [k, idx, _] = gt_kd.search_knn_vector_3d(seed_pts[j], args.patch_size * 2)
            gt_tmp = gt_pts[idx]
            training_set_gt.append(gt_pts[idx].tolist())
            low_res = gt_tmp[np.random.choice(len(gt_tmp), size=args.patch_size, replace=False)]
            
            training_set_input.append(low_res)
        
        
    tmp = np.asarray(training_set_gt)
    tmp2 = np.asarray(training_set_input)
    save_path = os.path.join(args.save_dir, "8i_poisson_8_poisson_16.h5")
    
    hf = h5py.File(save_path, 'w')
    
    hf.create_dataset('poisson_16', data=tmp)
    hf.create_dataset('poisson_8', data=tmp2)
    hf.close()