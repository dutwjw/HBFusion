import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from typing import List, Dict, Any, Tuple
from torch.utils.tensorboard import SummaryWriter
import random


from loss.base import BaseLoss
from loss.triplet import BatchTripletLoss
from loss.point import PointTripletLoss
from loss.bev import BevTripletLoss

class HBFusionLoss(BaseLoss):
    def __init__(self, batch_loss:Dict, point_loss:Dict, overlap_loss:Dict, bev_loss:Dict, point_loss_scale:float, overlap_loss_scale:float, bev_loss_scale:float):
        super().__init__()
        print("HBFusionLoss: point_loss_scale=%.2f overlap_loss_scale=%.2f"%(point_loss_scale, overlap_loss_scale))
        self.batch_loss = BatchTripletLoss(**batch_loss)
        self.point_loss = PointTripletLoss(**point_loss)
        self.bev_loss = BevTripletLoss(**bev_loss)
        self.point_loss_scale = point_loss_scale
        self.overlap_loss_scale = overlap_loss_scale
        self.bev_loss_scale = bev_loss_scale

    def __call__(self, 
        # model 
        embeddings:torch.Tensor, 
        coords:List[torch.Tensor], 
        feats:List[torch.Tensor], 
        # scores:List[torch.Tensor],
        bev_per_feature: torch.Tensor,
        # mask
        rotms:torch.Tensor, 
        trans:torch.Tensor,
        positives_mask:torch.Tensor, 
        negatives_mask:torch.Tensor,
        geneous:torch.Tensor
    ):
        # get global coords
        device, BS = embeddings.device, embeddings.shape[0]
        rotms, trans = rotms.to(device), trans.to(device)
        # R*p + T
        global_coords = [torch.mm(rotms[ndx], coords[ndx].clone().detach().transpose(0,1)).transpose(0,1) + trans[ndx].unsqueeze(0) for ndx in range(BS)]
        # compute point loss
        point_loss, point_stats = self.point_loss(feats, global_coords, positives_mask)

        batch_loss, batch_stats = self.batch_loss(embeddings, embeddings, positives_mask, negatives_mask)

        bev_loss, bev_stats = self.bev_loss(bev_per_feature)

        stats = {"batch":batch_stats, "point":point_stats, "bev":bev_stats}
        return batch_loss + self.point_loss_scale * point_loss + self.bev_loss_scale * bev_loss, stats

    def print_stats(self, epoch:int, phase:str, writer:SummaryWriter, stats:Dict[str, Any]):
        self.batch_loss.print_stats(epoch, phase, writer, stats["batch"])
        self.point_loss.print_stats(epoch, phase, writer, stats["point"])
        self.bev_loss.print_stats(epoch, phase, writer, stats["bev"])
        return