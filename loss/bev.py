import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from loss.base import BaseLoss
import matplotlib.pylab as plt


def get_max_per_row(mat: torch.Tensor, mask: torch.Tensor):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat: torch.Tensor, mask: torch.Tensor):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float("inf")
    return torch.min(mat_masked, dim=1), non_inf_rows


class BevTripletMiner:
    def __init__(self):
        return

    def __call__(self, dist_mat: torch.Tensor, positives_mask: torch.Tensor, negatives_mask: torch.Tensor):
        assert dist_mat.shape == positives_mask.shape == negatives_mask.shape
        with torch.no_grad():
            (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
            (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
            a_keep_idx = torch.where(a1p_keep & a2n_keep)[0]
            anc_ind = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
            pos_ind = hardest_positive_indices[a_keep_idx]
            neg_ind = hardest_negative_indices[a_keep_idx]

            stats = {
                "triplet_num": a_keep_idx.shape[0],
                "max_pos_dist": torch.max(hardest_positive_dist[a_keep_idx]).item(),
                "mean_pos_dist": torch.mean(hardest_positive_dist[a_keep_idx]).item(),
                "min_pos_dist": torch.min(hardest_positive_dist[a_keep_idx]).item(),
                "max_neg_dist": torch.max(hardest_negative_dist[a_keep_idx]).item(),
                "mean_neg_dist": torch.mean(hardest_negative_dist[a_keep_idx]).item(),
                "min_neg_dist": torch.min(hardest_negative_dist[a_keep_idx]).item(),
            }
            return anc_ind, pos_ind, neg_ind, stats


class BevTripletLoss(BaseLoss):
    def __init__(self, margin: float, style: str):
        super().__init__()
        assert style in ["soft", "hard"]
        print("BatchTripletLoss: margin=%.1f, style=%s" % (margin, style))
        self.miner = BevTripletMiner()
        self.margin = margin
        self.style = style
        return

    def __call__(self,
                 bev_per_feature: torch.Tensor,
                 ):
        stats = {}

        bev_feature_loss = \
            torch.norm(bev_per_feature[:, 0, :] - bev_per_feature[:, 1, :], dim = 1) +\
            torch.norm(bev_per_feature[:, 1, :] - bev_per_feature[:, 2, :], dim = 1) +\
            torch.norm(bev_per_feature[:, 2, :] - bev_per_feature[:, 3, :], dim = 1)

        loss = F.relu(bev_feature_loss + self.margin).mean()

        stats["loss"] = loss.item()
        return loss, stats

    def print_stats(self, epoch: int, phase: str, writer: SummaryWriter, stats: Dict[str, Any]):
        writer.add_scalar('Loss/BevLoss', stats["loss"], epoch)
        return
