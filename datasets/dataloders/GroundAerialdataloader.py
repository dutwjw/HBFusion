from torch.utils.data import DataLoader
from datasets.dataloders.samplers.batchsampler import BatchSampler
from datasets.dataloders.collates.GroundAerialCollate import GroundAerialCollate
from datasets.GroundAerialDataset import GroundAerialDataset


def GroundAerialDataLoader(**kw):
    """
    Create dataloaders
    """
    
    augment = None

    dataset = GroundAerialDataset(
        dataset=kw["dataset"], 
    )
    
    sampler = BatchSampler(
        dataset=dataset,
        **kw["sampler"]
    )

    collate = GroundAerialCollate(dataset=dataset, augment=augment, **kw["collate"])
    dataloader = DataLoader(
        dataset, 
        batch_sampler=sampler, 
        collate_fn=collate,
        num_workers=kw["num_workers"], 
        pin_memory=True
    )
    return dataloader
