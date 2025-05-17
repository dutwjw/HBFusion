import os
import torch
import glob
import numpy as np
from torch.utils.data import Dataset
from typing import List
from models.utils.bev.bev_feature import get_mul_bev
from torchsparse.utils.quantize import sparse_quantize


class GroundAerialDataset(Dataset):
    """
    # Dataset wrapper for GroundAerialDataset
    """
    def __init__(self, dataset:dict,):
        self.rootpath = dataset['root_path']

        if 'sample_proportion' in dataset:
            self.sample_proportion = dataset['sample_proportion']
        else:
            self.sample_proportion = 1.0
        assert os.path.exists(self.rootpath), "Cannot access rootpath {}".format(self.rootpath)
        print("GroundAerialDataset: {}".format(self.rootpath))
        # 0: ground, 1: aerial
        self.geneous_names = ["ground", "aerial"]
        self.Ng = len(self.geneous_names)
        self.geneous = np.load(os.path.join(self.rootpath, "geneous.npy"))
        self.Nm = self.geneous.shape[0]
        
        self.homoindices = [[] for _ in self.geneous_names]
        for ndx in range(self.Nm):
            self.homoindices[self.geneous[ndx]].append(ndx)
        self.homoindices = [np.asarray(e) for e in self.homoindices]

        # tum format (Nm, 8) [t, x, y, z, qx, qy, qz, qw]
        self.tum = np.load(os.path.join(self.rootpath, "tum.npy"))
        assert self.Nm == self.tum.shape[0], "GroundAerialDataset: self.Nm != self.tum.shape[0]"

        # make self check files
        self.checkpath = os.path.join(self.rootpath, "selfcheck")
        if not os.path.exists(self.checkpath): os.mkdir(self.checkpath)

        self.anchors:np.ndarray = None
        
        # load data 
        self.pcs           = [os.path.join(self.rootpath, "items", "%06d" % ndx, "pointcloud.npy")  for ndx in range(self.Nm)]
        self.positives     = [os.path.join(self.rootpath, "items", "%06d"%ndx, "positives.npy")     for ndx in range(self.Nm)]
        self.non_negatives = [os.path.join(self.rootpath, "items", "%06d"%ndx, "non_negatives.npy") for ndx in range(self.Nm)]
        self.get_anchors()

    def __len__(self):
        return self.Nm

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        pc_ = self.get_pc(ndx)

        _, index_ = sparse_quantize(pc_, 0.6, return_index=True)
        pc_ = pc_[index_]

        # multiple point density
        pc_ = self.samplePcByRatio(pc_)

        bevs = get_mul_bev(pc_)
        
        pc = torch.tensor(pc_)
        
        return ndx, pc, bevs

    def samplePcByRatio(self, sourcePc):
        tmp = np.arange(sourcePc.shape[0])
        np.random.shuffle(tmp)
        return sourcePc[tmp[:int(self.sample_proportion * sourcePc.shape[0])], :]

    def get_indices(self) -> np.ndarray:
        return np.arange(self.Nm)
    
    def get_homoindices(self, geneous_id:int) -> np.ndarray:
        return np.copy(self.homoindices[geneous_id])

    def get_geneous_names(self) -> List[str]:
        return self.geneous_names

    def get_all_geneous(self) -> np.ndarray:
        return np.copy(self.geneous)

    def get_positives(self, ndx:int) -> np.ndarray:
        return np.load(self.positives[ndx])
    
    def get_non_negatives(self, ndx:int) -> np.ndarray:
        return np.load(self.non_negatives[ndx])

    def get_tum(self, ndx:int):
        return np.copy(self.tum[ndx])

    def get_correspondences(self, source_ndx:int, target_ndx:int)  -> np.ndarray:
        path = os.path.join(
            self.rootpath, 
            "items", 
            "%06d"%source_ndx, 
            "correspondence",
            "%06d.npy"%target_ndx
        )
        return np.load(path)

    def get_pc(self, ndx) -> np.ndarray:
        return np.load(self.pcs[ndx])

    def get_mul_bev_path(self, ndx):
        files =  self.get_files(os.path.join(self.rootpath, 'multi_bev_img', f'img{ndx}'), 'png')
        files.sort(key=lambda x:int(x.split('/')[-1].split('.')[0][3:]))
        return files

    def get_files(self, path, extension):
        return glob.glob(os.path.join(path, f'*{extension}'))

    def get_anchors(self) -> np.ndarray:
        """
        # Get indices of items with heterogeneous positive samples in dataset
        """
        if self.anchors is not None: return np.copy(self.anchors)
        print("GroundAerialDataset: self.anchors is None, generating")
        anchors = []
        for i in self.get_indices():
            positives = self.get_positives(i)
            is_anchor = True
            for gid, gname in enumerate(self.geneous_names):
                if np.intersect1d(positives, self.get_homoindices(gid)).shape[0] == 0:
                    is_anchor = False
                    break
            if is_anchor: anchors.append(i)

        self.anchors = np.asarray(anchors)

        for gid, gname in enumerate(self.get_geneous_names()):
            ganchors = np.intersect1d(
                self.anchors,
                self.get_homoindices(gid)
            ).shape[0]
            print("GroundAerialDataset: %s has %d anchors" % (gname, ganchors))
        
        return np.copy(self.anchors)
