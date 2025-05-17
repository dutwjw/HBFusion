import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from typing import List, Dict, Any
from loss.base import BaseLoss
from torch.utils.tensorboard import SummaryWriter

def get_max_per_row(mat:torch.Tensor, mask:torch.Tensor):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat:torch.Tensor, mask:torch.Tensor):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float("inf")
    return torch.min(mat_masked, dim=1), non_inf_rows


class PointTripletLoss(BaseLoss):
    def __init__(self, margin:float, style:str, corr_dist:float, sample_num:int, pos_dist:float, neg_dist:float):
        super().__init__()
        assert style in ["soft", "hard"]
        print("PointTripletLoss: margin=%.1f, style=%s" % (margin, style))
        self.margin     = margin
        self.style      = style
        self.corr_dist  = corr_dist
        self.sample_num = sample_num
        self.pos_dist   = pos_dist
        self.neg_dist   = neg_dist

    def __call__(self, feats:List[torch.Tensor], coords:List[torch.Tensor], positives_mask:torch.Tensor):
        source_indices, target_indices = torch.where(positives_mask == True)
        select_indices = torch.where(source_indices < target_indices)
        source_indices, target_indices = source_indices[select_indices].tolist(), target_indices[select_indices].tolist()
        
        losses = []
        point_stats = {
            "fitness":[],
            "triplet_num":[],
            "non_zero_triplet_num":[],
            "pos_min":[], 
            "pos_mean":[], 
            "pos_max":[], 
            "neg_min":[],
            "neg_mean":[], 
            "neg_max":[]
        }
        
        for sndx, tndx in zip(source_indices, target_indices):
            # construct pcd from coords
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(coords[sndx].clone().detach().cpu().numpy())
            source_pcd.paint_uniform_color([0,0,1])
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(coords[tndx].clone().detach().cpu().numpy())
            target_pcd.paint_uniform_color([1,0,0])
            # o3d.visualization.draw_geometries([source_pcd, target_pcd])
            # icp and get points set
            reg_p2p = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, self.corr_dist, np.eye(4))
            corr_set = np.asarray(reg_p2p.correspondence_set)
            assert reg_p2p.fitness > 0.05 
            # sample, Ns = 64
            sample_indices = np.random.choice(corr_set.shape[0], min(corr_set.shape[0], self.sample_num))
            Ns = sample_indices.shape[0]
            # sample_set: 
            # [ [s0, s1, s2, ... ],   Ns
            #   [t0, t1, t2, ... ] ]  Ns
            sample_set = corr_set[sample_indices].T.tolist()

            # sample coords and feats
            scoord, tcoord = coords[sndx][sample_set[0]], coords[tndx][sample_set[1]]
            sfeat,  tfeat  = feats[sndx][sample_set[0]],  feats[tndx][sample_set[1]]
            # Ns * Ns
            coord_dist = torch.norm(scoord.unsqueeze(1) - tcoord.unsqueeze(0), dim=2)
            # Ns * Ns
            feat_dist = torch.norm(sfeat.unsqueeze(1) - tfeat.unsqueeze(0), dim=2)
            # get hardest positive and negative
            (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(feat_dist, coord_dist < self.pos_dist)
            (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(feat_dist, coord_dist > self.neg_dist)
            # positive <=> anchor <=> negative
            a_keep_idx = torch.where(a1p_keep & a2n_keep)[0]
            triplet_num = a_keep_idx.shape[0]
            if triplet_num == 0: continue

            anc_ind = torch.arange(Ns).to(hardest_positive_indices.device)[a_keep_idx]
            pos_ind = hardest_positive_indices[a_keep_idx]
            neg_ind = hardest_negative_indices[a_keep_idx]

            triplet_dist = torch.norm(sfeat[anc_ind] - tfeat[pos_ind], dim=1) - torch.norm(sfeat[anc_ind] - tfeat[neg_ind], dim=1)
            
            non_zero_triplet_num = torch.where((triplet_dist + self.margin) > 0)[0].shape[0]

            
            if self.style == "hard":
                this_pair_loss = F.relu(triplet_dist + self.margin).mean()
            elif self.style == "soft":
                this_pair_loss = torch.log(1+self.margin*torch.exp(triplet_dist)).mean()
            else:
                raise NotImplementedError(f"PointTripletLoss: unkown style {self.style}")
            # this_pair_loss = F.relu(triplet_dist).mean()

            losses.append(this_pair_loss)

            point_stats["fitness"].append(reg_p2p.fitness)
            point_stats["triplet_num"].append(triplet_num)
            point_stats["non_zero_triplet_num"].append(non_zero_triplet_num)
            point_stats["pos_min"].append(hardest_positive_dist[a_keep_idx].min().item())
            point_stats["pos_mean"].append(hardest_positive_dist[a_keep_idx].mean().item())
            point_stats["pos_max"].append(hardest_positive_dist[a_keep_idx].max().item())
            point_stats["neg_min"].append(hardest_negative_dist[a_keep_idx].min().item())
            point_stats["neg_mean"].append(hardest_negative_dist[a_keep_idx].mean().item())
            point_stats["neg_max"].append(hardest_negative_dist[a_keep_idx].max().item())
        
        avg_point_stats = {e: np.mean(point_stats[e]) for e in point_stats}
        loss = torch.stack(losses).mean()
        avg_point_stats["loss"] = loss.item()

        return loss, avg_point_stats

    def print_stats(self, epoch:int, phase:str, writer:SummaryWriter, stats:Dict[str, Any]):
        writer.add_scalar('Loss/PointTripletLoss', stats["loss"], epoch)
        return 
