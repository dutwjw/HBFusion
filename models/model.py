import torch
import os
import copy
import yaml
from typing import Dict, Any

class Model:
    """
    # Wrapper for models
    """ 
    def __init__(self):
        return

    def construct(self, name:str, **kw):
        self.config = copy.deepcopy(kw)
        self.config["name"] = name
        self.name = name
        self.model = None
        if self.name == "HBFusion":
            from models.hbfusion import HBFusion
            self.model = HBFusion(**kw)
        else:
            raise NotImplementedError("Model: model %s not implemented" % self.name)

    def __call__(self, data:Dict[str, Any]) -> Dict[str, Any]:
        output:Dict[str, Any] = {}
        if self.name == "HBFusion":
            assert set(["coords", "feats", "geneous", ]) <= set(data.keys())
            output["coords"], output["feats"], output["embeddings"], output["bev_per_feature"] = self.model(data["coords"], data["feats"],  data["geneous"], data["bevs"])
        else:
            raise NotImplementedError("Model: model %s not implemented" % self.name)
        return output
    
    def save(self, path:str):
        pth_file_dict = {"config":self.config, "weight": self.model.state_dict()}
        torch.save(pth_file_dict, path)
        return

    def load(self, path, device):
        pth_file_dict = torch.load(path, map_location=device)
        print("Model: load\n", pth_file_dict["config"])
        self.construct(**pth_file_dict["config"])
        self.model.load_state_dict(pth_file_dict["weight"])
        return
