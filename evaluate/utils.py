import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import DataLoader
from misc.utils import tensors2device
from time import time

def get_embeddings(model, dataloader: DataLoader, device:str, print_stats:bool=True):
    model.model = model.model.to(device)
    model.model.eval()
    embeddings = []
    if print_stats: iterater = tqdm(dataloader, desc="Getting embedding")
    else: iterater=dataloader
    for data, mask in iterater:
        data = tensors2device(data, device)
        with torch.no_grad():
            t1 = time()
            output = model(data)
            assert "embeddings" in output, "Evaluate: no embeddings in model output"
            embeddings.append(output["embeddings"].clone().detach().cpu().numpy().reshape(1,-1))

    embeddings = np.concatenate(embeddings, axis=0)
    if print_stats: print("Embeddings size = ", embeddings.shape)
    return embeddings

def get_recall_precision_curve( 
    dataset,
    distance:np.ndarray, 
    source_indices:np.ndarray, 
    target_indices:np.ndarray,
    num_eval:int,
):
    rp = np.empty((0,2))
    ds = np.linspace(np.min(distance)-0.01, np.max(distance)+0.01, num_eval)

    for threshold in tqdm(ds):
        threshold_rp = np.empty((0,2))
        for i in source_indices:

            real_positive = np.intersect1d(
                dataset.get_positives(i),
                target_indices,
            )
            
            pred_positive = np.intersect1d(
                np.where(distance[i] < threshold)[0],
                target_indices,
            )
            
            if real_positive.shape[0] == 0: continue
            
            tp = np.intersect1d(real_positive, pred_positive).shape[0] # actual is truth (t) and predit is positive (p)
            fn = np.setdiff1d(real_positive, pred_positive).shape[0]
            fp = np.setdiff1d(pred_positive, real_positive).shape[0]
            recall, precision = 0., 0.
            if tp == 0:
                if   fn == 0 and fp == 0: continue
                elif fn == 0 and fp != 0: recall, precision = 1., 0.
                elif fn != 0 and fp == 0: recall, precision = 0., 1.
                else:                     recall, precision = 0., 0.
            else:
                recall = float(tp)/float(tp+fn)
                precision = float(tp)/float(tp+fp)

            threshold_rp = np.concatenate([threshold_rp, np.asarray([[recall, precision]])], axis=0)
        
        if threshold_rp.shape[0] == 0: continue
        threshold_rp = np.mean(np.asarray(threshold_rp), axis=0)
        
        rp = np.concatenate([rp, threshold_rp.reshape(1,2)], axis=0)
    # [N, 2] -> [2, N] 
    rp = rp.T
    indices = np.argsort(rp[0])
    rp = rp[:, indices]
    ds = ds[indices]
    return rp, ds

def get_hetero_recall_precision(
    dataset,
    distance:np.ndarray,
    savepath:str=None,
    num_eval:int=100,
    show:bool=False
):
    
    assert len(dataset) == distance.shape[0], "Evaluate: len(datasets) == embeddings.shape[0]"
    Nm = distance.shape[0]
    distance = distance.copy() + np.eye(Nm)*(np.max(distance)+0.01)

    all_rp = {}
    
    geneous_names = dataset.get_geneous_names()
    print('Get hetero recall precision')
    for sgid, source in enumerate(geneous_names):
        sgndx_all = dataset.get_homoindices(sgid)
        for tgid, target in enumerate(geneous_names):
            tgndx_all = dataset.get_homoindices(tgid)
            st = "{}-{}".format(source, target)

            print(st)
            if sgndx_all.shape[0] == 0 or tgndx_all.shape[0] == 0:
                print("no instance in source or target, continue")
                continue
            # recall-pricision
            all_rp[st] = {}
            all_rp[st]["xy"], all_rp[st]["ds"] = get_recall_precision_curve(dataset, distance, sgndx_all, tgndx_all, num_eval)

    return all_rp

def show_closest(dataset, distance:np.ndarray):
    print("Show Closest Submaps")

    geneous_names = dataset.get_geneous_names()
    for sgid, source in enumerate(geneous_names):
        sgndx_all = dataset.get_homoindices(sgid)
        for tgid, target in enumerate(geneous_names):
            if source == target: continue

            tgndx_all = dataset.get_homoindices(tgid)
            for _ in range(10):
                d = 4.0
                anchor, closest = None, None
                while d > 2.0:
                    anchor = random.choice(sgndx_all)
                    closest = tgndx_all[np.argsort(distance[anchor, tgndx_all])][0]
                    d = distance[anchor][closest]
                    

                suc = "False"
                if closest in dataset.get_positives(anchor): suc = "True"
                
                anchor_pcd = o3d.geometry.PointCloud()
                anchor_pcd.points = o3d.utility.Vector3dVector(dataset.get_pc(anchor) - np.asarray([40,0,0]))
                
                closest_pcd = o3d.geometry.PointCloud()
                closest_pcd.points = o3d.utility.Vector3dVector(dataset.get_pc(closest) + np.asarray([40,0,0]))

                o3d.visualization.draw_geometries(
                    [anchor_pcd, closest_pcd], 
                    window_name="%s-%s: result=%s, distance=%.3f"%(source, target, suc, d)
                )
