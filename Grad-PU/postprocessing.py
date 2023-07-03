import open3d as o3d

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("/home/v-chendwang/test-playground/Grad-PU/data/8i/longdress/input_2X/results/upsample_ld/longdress_vox10_1196.xyz")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
    o3d.io.write_point_cloud("/home/v-chendwang/test-playground/Grad-PU/data/8i/longdress/input_2X/results/upsample_ld/longdress_vox10_1196_inlier.xyz", cl)