import torch
import torchsparse.nn.functional as F_sparse
import numpy as np
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets
from torchsparse.utils.collate import sparse_collate
import torchsparse
import torch.nn as nn

class SparseConv3d(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            torchsparse.nn.Conv3d(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride), torchsparse.nn.BatchNorm(outc),
            torchsparse.nn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class SparseDeConv3d(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            torchsparse.nn.Conv3d(inc,
                                 outc,
                                 kernel_size=ks,
                                 stride=stride,
                                 transposed=True), torchsparse.nn.BatchNorm(outc),
            torchsparse.nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class SparseConv3dRes(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            torchsparse.nn.Conv3d(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride), torchsparse.nn.BatchNorm(outc),
            torchsparse.nn.ReLU(True),
            torchsparse.nn.Conv3d(outc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=1), torchsparse.nn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                torchsparse.nn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                torchsparse.nn.BatchNorm(outc)
            )

        self.relu = torchsparse.nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

def map_coords(x):
    x -= np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1
    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h

def voxelization(lidar_pc, voxel_size):
    coords = np.round(lidar_pc[:, :3] / voxel_size)
    coords -= coords.min(0, keepdims=1)
    feats = lidar_pc
    coords_ = np.floor(coords).astype(np.int32)
    _, indices, _ = np.unique(map_coords(coords_), return_index=True, return_inverse=True)
    coords = coords[indices]
    feats = feats[indices]
    inputs = SparseTensor(feats, coords)
    inputs = sparse_collate([inputs])
    inputs.C = inputs.C.int()
    return inputs, feats

def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1)

    pc_hash = F_sparse.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F_sparse.sphashquery(pc_hash, sparse_hash)
    counts = F_sparse.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = F_sparse.spvoxelize(torch.floor(new_float_coord), idx_query,
                                   counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F_sparse.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor

def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get('idx_query') is None\
       or z.additional_features['idx_query'].get(x.s) is None:
        pc_hash = F_sparse.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F_sparse.sphash(x.C)
        idx_query = F_sparse.sphashquery(pc_hash, sparse_hash)
        counts = F_sparse.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F_sparse.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor

def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = F_sparse.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = F_sparse.sphash(x.C.to(z.F.device))
        idx_query = F_sparse.sphashquery(old_hash, pc_hash)
        weights = F_sparse.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = F_sparse.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = F_sparse.spdevoxelize(x.F, z.idx_query.get(x.s),
                                  z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor

class SparseConvolution(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cs = [16, 32, 64, 128, 256, 256, 128, 64, 32]

        self.stem = nn.Sequential(
            torchsparse.nn.Conv3d(3, cs[0], kernel_size=3, stride=1),
            torchsparse.nn.BatchNorm(cs[0]), torchsparse.nn.ReLU(True),
        )

        self.stage1 = nn.Sequential(
            SparseConv3d(cs[0], cs[0], ks=2, stride=2, dilation=1),
            SparseConv3dRes(cs[0], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            SparseConv3d(cs[1], cs[1], ks=2, stride=2, dilation=1),
            SparseConv3dRes(cs[1], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            SparseConv3d(cs[2], cs[2], ks=2, stride=2, dilation=1),
            SparseConv3dRes(cs[2], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            SparseConv3d(cs[3], cs[3], ks=2, stride=2, dilation=1),
            SparseConv3dRes(cs[3], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            SparseDeConv3d(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                SparseConv3dRes(cs[5] + cs[3], cs[5], ks=3, stride=1,
                              dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            SparseDeConv3d(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                SparseConv3dRes(cs[6] + cs[2], cs[6], ks=3, stride=1,
                              dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            SparseDeConv3d(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                SparseConv3dRes(cs[7] + cs[1], cs[7], ks=3, stride=1,
                              dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            SparseDeConv3d(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                SparseConv3dRes(cs[8] + cs[0], cs[8], ks=3, stride=1,
                              dilation=1),
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8], 32))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[1], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[2], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[3], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[5], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
        ])

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.dropout = nn.Dropout(0.3, True)

    def forward(self, x):
        pt = PointTensor(x.F, x.C.float())

        vt = initial_voxelize(pt, 0.05, 0.05)

        vt = self.stem(vt)
        pt0 = voxel_to_point(vt, pt, nearest=False)
        pt0.F = pt0.F
        dsFeature1 = point_to_voxel(vt, pt0)
        dsFeature1 = self.stage1(dsFeature1)
        dsFeature2 = self.stage2(dsFeature1)
        dsFeature3 = self.stage3(dsFeature2)
        dsFeature4 = self.stage4(dsFeature3)
        
        upFeature1 = self.up1[0](dsFeature4)
        upFeature1 = torchsparse.cat([upFeature1, dsFeature3])
        upFeature1 = self.up1[1](upFeature1)
        upFeature2 = self.up2[0](upFeature1)
        upFeature2 = torchsparse.cat([upFeature2, dsFeature2])
        upFeature2 = self.up2[1](upFeature2)
        upFeature3 = self.up3[0](upFeature2)
        upFeature3 = torchsparse.cat([upFeature3, dsFeature1])
        upFeature3 = self.up3[1](upFeature3)
        upFeature4 = self.up4[0](upFeature3)
        upFeature4 = torchsparse.cat([upFeature4, vt])
        upFeature4 = self.up4[1](upFeature4)
        
        pt1 = voxel_to_point(dsFeature4, pt0)
        pt2 = voxel_to_point(upFeature1, pt0)
        pt3 = voxel_to_point(upFeature2, pt0)
        pt4 = voxel_to_point(upFeature3, pt0)
        pt5 = voxel_to_point(upFeature4, pt0)
        pt5.F = \
            self.point_transforms[1](pt5.F) + \
            self.point_transforms[2](pt4.F) + \
            self.point_transforms[3](pt3.F) + \
            self.point_transforms[4](pt2.F) + \
            self.point_transforms[5](pt1.F)
        
        # split the feature [N_sum, D] to [B, N, D]
        last_col = pt5.C[:, -1].int()
        _, counts = last_col.unique(return_counts=True)
        split_tensors = list(torch.split(pt5.F, counts.tolist()))

        return split_tensors

