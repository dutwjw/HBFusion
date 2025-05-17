from typing import List
from torch.utils.data import Sampler
from datasets.GroundAerialDataset import GroundAerialDataset
class BatchSampler(Sampler[List[int]]):
    """
    # Wrapper for all sampler
    """
    def __init__(
        self, 
        dataset:GroundAerialDataset, 
        batch_size:int, 
        batch_size_limit:int,
        batch_expansion_rate:float, 
        **kw,
    ):
        # sample factory
        self.sample_fn = None
        from datasets.dataloders.samplers.hetero import HeteroTripletSample
        self.sample_fn = HeteroTripletSample(dataset=dataset, **kw)

        self.batch_size = batch_size - batch_size%self.sample_fn.get_k()
        self.batch_size_limit = batch_size_limit
        self.batch_expansion_rate = batch_expansion_rate
        self.batch_idx = []
        

    def __iter__(self):
        """
        # Generate A Bacth_idx
        """
        self.batch_idx = self.sample_fn(self.batch_size)

        for batch in self.batch_idx: yield batch

    def __len__(self):
        return len(self.batch_idx)
