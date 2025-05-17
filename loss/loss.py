from typing import Any, Dict
from misc.utils import tensors2numbers
from torch.utils.tensorboard import SummaryWriter

class Loss:
    """
    # Wrapper loss function 
    """
    def __init__(self, name:str, **kw):
        self.name = name
        if self.name == "HBFusionLoss":
            from loss.hbfusion import HBFusionLoss
            self.loss_fn = HBFusionLoss(**kw)
        else:
            raise NotImplementedError("HBFusionLoss: loss_fn %s not implemented" % self.name)

    def __call__(self, output:Dict[str, Any], mask:Dict[str, Any]):
        loss, stats = None, None
        if self.name == "HBFusionLoss":
            loss, stats = self.loss_fn(
                output["embeddings"], 
                output["coords"], 
                output["feats"], 
                output["bev_per_feature"],
                mask["rotms"],
                mask["trans"],
                mask["positives"], 
                mask["negatives"],
                mask["geneous"],
            ) 
        else:
            raise NotImplementedError("HBFusionLoss: loss_fn %s not implemented" % self.name)
        
        assert loss is not None and stats is not None
        stats = tensors2numbers(stats)
        return loss, stats
    
    def print_stats(self, epoch:int, phase:str, writer:SummaryWriter, stats:Dict[str, Any]):
        """
        # visualize stats
        """
        if self.name == "HBFusionLoss":
            self.loss_fn.print_stats(epoch, phase, writer, stats)
        else:
            raise NotImplementedError("HBFusionLoss: loss_fn %s.print_stats() not implemented" % self.name)