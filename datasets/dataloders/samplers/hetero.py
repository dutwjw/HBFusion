import random
import numpy as np
from typing import List
from copy import deepcopy
from datasets.GroundAerialDataset import GroundAerialDataset
from datasets.dataloders.samplers.base import BaseSample

class HeteroTripletSample(BaseSample):
    def __init__(self, dataset:GroundAerialDataset, max_batches:int):
        """
        # Sampling mechanism for heterogeneous data
        """
        print("Sampling Mechanism: HeteroTripletSample")
        self.dataset = dataset
        self.max_batches = max_batches
        self.k = 2
        
    def get_k(self) -> int:
        """
        # Ensure batch_size % k == 0
        """
        return self.k
    
    def __call__(self, batch_size:int) -> List[List[int]]:

        # batches indices of an training epoch
        batch_idx:List[List[int]] = []
        # unused 
        unused_elements_ndx:List[int] = self.dataset.get_indices().tolist()
        # current
        current_batch:List[int] = []

        # items with heterogeneous positive samples in dataset
        anchors:List[int] = self.dataset.get_anchors().tolist()

        while True:
            anchor = random.choice(anchors)
            anchor_geneous = self.dataset.get_all_geneous()[anchor]
            current_batch.append(anchor)
            anchors.remove(anchor)
            unused_elements_ndx.remove(anchor)

            unused_elements_ndx_np = np.asarray(unused_elements_ndx)
            positives = self.dataset.get_positives(anchor)

            for gid, _ in enumerate(self.dataset.get_geneous_names()):
                if gid == anchor_geneous: continue
                geneous_positives = np.intersect1d(positives, self.dataset.get_homoindices(gid))
                unused_geneous_positives = np.intersect1d(
                    unused_elements_ndx_np,
                    geneous_positives
                )
                this_geneous_positive:int=None
                if len(unused_geneous_positives) != 0:
                    this_geneous_positive = random.choice(unused_geneous_positives.tolist())
                    unused_elements_ndx.remove(this_geneous_positive) 
                else:
                    this_geneous_positive = random.choice(geneous_positives.tolist())
                
                current_batch.append(this_geneous_positive)
                 
                if this_geneous_positive in anchors: anchors.remove(this_geneous_positive)

            if len(current_batch) >= batch_size:
                assert len(current_batch) % self.k == 0
                batch_idx.append(deepcopy(current_batch))
                current_batch = []
                if (self.max_batches is not None) and (len(batch_idx) >= self.max_batches):
                    break

            if len(unused_elements_ndx) == 0 or len(anchors) == 0:
                break
        return batch_idx
