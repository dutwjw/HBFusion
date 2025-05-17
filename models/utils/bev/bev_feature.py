import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os, glob, sys, torch, math
from tqdm import trange, tqdm
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from models.utils.bev.NetVLAD import NetVLAD
from models.utils.aggregation.gem import MeanGeM_mean_dim_one
from time import time

def get_files(path, extension):
    return glob.glob(os.path.join(path, f'*{extension}'))


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def getBEV(all_points):
    # BEV picture parameter
    x_min = -30
    y_min = -30
    x_max = 30
    y_max = 30
    sample_size = 0.3

    all_points_pc = o3d.geometry.PointCloud()
    all_points_pc.points = o3d.utility.Vector3dVector(all_points)
    all_points_pc = all_points_pc.voxel_down_sample(voxel_size=sample_size)
    all_points = np.asarray(all_points_pc.points)

    x_min_ind = np.floor(x_min / sample_size).astype(int)
    x_max_ind = np.floor(x_max / sample_size).astype(int)
    y_min_ind = np.floor(y_min / sample_size).astype(int)
    y_max_ind = np.floor(y_max / sample_size).astype(int)

    x_num = x_max_ind - x_min_ind + 1
    y_num = y_max_ind - y_min_ind + 1

    mat_global_image = np.zeros((x_num, y_num), dtype=np.uint8)

    for i in range(all_points.shape[0]):
        x_ind = x_max_ind - np.floor(all_points[i, 0] / sample_size).astype(int)
        y_ind = y_max_ind - np.floor(all_points[i, 1] / sample_size).astype(int)
        if (x_ind >= x_num or y_ind >= y_num):
            continue
        if mat_global_image[x_ind, y_ind] < 2:
            mat_global_image[x_ind, y_ind] += 1

    max_pixel = np.max(mat_global_image)

    mat_global_image = mat_global_image / max_pixel * 255

    return (mat_global_image / 255.0).astype(np.float32)


def get_mul_bev(all_points):
    x_min = -30
    y_min = -30
    x_max = 30
    y_max = 30
    sample_size = 1

    all_points_pc = o3d.geometry.PointCloud()
    all_points_pc.points = o3d.utility.Vector3dVector(all_points)
    all_points_pc = all_points_pc.voxel_down_sample(voxel_size=sample_size)

    _, ind = all_points_pc.remove_statistical_outlier(nb_neighbors=10, std_ratio=5.0)
    all_points_pc = all_points_pc.select_by_index(ind)

    all_points_np = np.array(all_points_pc.points)
    z_value = all_points_np[:, 2]
    low_indices = np.argsort(z_value)[:10]
    low_points = all_points_np[low_indices, 2][-1] + 0.5
    all_points_np[:, 2] -= low_points
    all_points_np = all_points_np[all_points_np[:, 2] >= 0]

    # generate multi bevs
    x_min_ind = np.floor(x_min / sample_size).astype(int)
    x_max_ind = np.floor(x_max / sample_size).astype(int)
    y_min_ind = np.floor(y_min / sample_size).astype(int)
    y_max_ind = np.floor(y_max / sample_size).astype(int)

    x_num = x_max_ind - x_min_ind + 1
    y_num = y_max_ind - y_min_ind + 1

    layers_num = 4 
    layers_height = 4
    bev_pix_max_count = 1

    mat_global_image = np.zeros((layers_num, x_num, y_num), dtype=np.uint8)

    x_ind = x_max_ind - np.floor(all_points_np[:, 0] / sample_size).astype(int)
    y_ind = y_max_ind - np.floor(all_points_np[:, 1] / sample_size).astype(int)
    bev_layer_indice = np.floor(all_points_np[:, 2] / layers_height).astype(int)
    x_ind_where = np.where(x_ind < x_num)[0]
    y_ind_where = np.where(y_ind < y_num)[0]
    ind_final = np.union1d(x_ind_where, y_ind_where)
    x_ind = x_ind[ind_final]
    y_ind = y_ind[ind_final]
    bev_layer_indice = bev_layer_indice[ind_final]
    bev_layer_indice = np.clip(bev_layer_indice, a_min=None, a_max=3)
    mat_global_image[bev_layer_indice, x_ind, y_ind] = 1

    for i, mat_global_image_single_layer in enumerate(mat_global_image):
        max_pixel = np.max(mat_global_image_single_layer)
        if max_pixel != 0:
            mat_global_image[i] = mat_global_image_single_layer / max_pixel * 255

    return (mat_global_image / 255.0).astype(np.float32)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: bool = False,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
    ) -> None:
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.ds_layer = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.ds_layer(x)

        out += identity
        out = self.relu(out)

        return out


class Multi_BEV(nn.Module):
    def __init__(self):
        super(Multi_BEV, self).__init__()

        # NetVLAD
        self.pool = NetVLAD()
        self.embedding_feature = 32
        # cnn backbone
        self.encoder = nn.Sequential(
            BasicBlock(1, 16, downsample=True),
            BasicBlock(16, 16, downsample=False),
            BasicBlock(16, 16, downsample=False),
            BasicBlock(16, self.embedding_feature, downsample=True),
            BasicBlock(self.embedding_feature, self.embedding_feature, downsample=False),
            BasicBlock(self.embedding_feature, self.embedding_feature, downsample=False),
        )
        
        self.score_fn = nn.Sequential(
            nn.Linear(self.embedding_feature, self.embedding_feature, bias=False),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        B, bevs_num, H, W, _ = x.shape
        x = x.reshape(-1, H, W, 1).permute(0, 3, 1, 2) 
        x = self.encoder(x)
        x = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * x
        x = self.pool(x) 
        x = x.reshape(B, bevs_num, -1)
        out = torch.sum(x, dim = 1, keepdim=True).squeeze(1)

        return x, out
