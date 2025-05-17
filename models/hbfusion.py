import torch
import torch.nn as nn
from typing import Dict

from models.utils.aggregation.gem import MeanGeM
from models.utils.bev.bev_feature import Multi_BEV
import torch.nn.functional as F
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate
from torchsparse import SparseTensor
from models.utils.sparseconvolution.sparse_convolution_pipeline import SparseConvolution

class HBFusion(nn.Module):
    def __init__(self, meangem:Dict, **kw):
        super(HBFusion, self).__init__()

        self.geneous_names = ["ground", "aerial"]
        self.meangem = MeanGeM(**meangem)
        self.mulBEV = Multi_BEV()
        self.linear = nn.Linear(160, 256)
        self.bn = nn.BatchNorm1d(256)
        self.sparse_convolution = SparseConvolution()

    def forward(self, coords: torch.Tensor, feats: torch.Tensor, geneous: torch.Tensor, bevs:torch.Tensor):
        BS = geneous.shape[0]
 
        # generate data used for torch sparse convolution
        coordsSparseTensor_list = []
        pc_list = []

        # genertate sparse tensor every batch
        for ndx in range(BS):
            pc = coords[ndx].float() #
            coords_, indices = sparse_quantize(pc.cpu().numpy(), 0.6, return_index=True) # voxelization
            coords_ = torch.tensor(coords_, dtype=torch.int, device=pc.device) # genertate sparse tensor
            feats_ = pc[indices]
            coordsSparseTensor_list.append(SparseTensor(coords=coords_, feats=feats_))
            pc_list.append(pc[indices]) # save the voxelized point clouds

        feature = self.sparse_convolution(sparse_collate(coordsSparseTensor_list))

        bev_per_feature, bev_sum_feature = self.mulBEV(bevs)

        batch_feats = torch.stack([self.meangem(feat) for feat in feature], dim=0)

        concate_feats = torch.cat([bev_sum_feature, batch_feats], 1)
        batch_feats = F.relu(self.bn(self.linear(concate_feats)))

        return pc_list, feature, batch_feats, bev_per_feature
