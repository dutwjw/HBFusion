import os
import sys
import argparse
import yaml
import torch
from tqdm import tqdm
from typing import Dict, List
import shutil

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from datasets.dataloders.GroundAerialdataloader import GroundAerialDataLoader
from models.model import Model
from loss.loss import Loss
from misc.utils import get_datetime, tensors2device, avg_stats

from torch.utils.tensorboard import SummaryWriter


def parse_opt()->dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml",       default='config/train/train.yaml', type=str, required=False)

    opt = parser.parse_args()
    opt = vars(opt)
    f = open(opt["yaml"], encoding="utf-8")
    train = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return train

def main(**kw):
    local_rank = 0
    torch.cuda.set_device(local_rank)
    
    # get dataloders
    dataloaders = {phase: GroundAerialDataLoader(**kw["dataloaders"]["train"]) for phase in kw["dataloaders"]}
    # get model
    model = Model()
    model.construct(**kw["method"]["model"])
    # get loss function
    loss_fn  = Loss(**kw["method"]["loss"])
    # model to local_rank
    model.model = model.model.to(local_rank)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=kw["train"]["lr"], weight_decay=kw["train"]["weight_decay"])
    # get scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, kw["train"]["scheduler_milestones"], gamma=0.1)

    # set results
    writer, weights_path = None, None
        
    model_name = get_datetime()
    print(f'TIME is {model_name}')
    if kw["results"]["weights"] is not None:
        weights_path = os.path.join(kw["results"]["weights"], model_name)
        if not os.path.exists(weights_path): os.mkdir(weights_path)
        # save config yaml
        with open(os.path.join(weights_path, "config.yaml"), "w") as file:
            file.write(yaml.dump(dict(kw), allow_unicode=True))
    if kw["results"]["logs"] is not None:
        logs_path = os.path.join(kw["results"]["logs"], model_name)
        writer = SummaryWriter(logs_path)

    # copy model and dataset to 'results/weights'
    if not os.path.exists(os.path.join(os.getcwd(), weights_path, 'datasets')): os.mkdir(os.path.join(os.getcwd(), weights_path, 'datasets'))
    if not os.path.exists(os.path.join(os.getcwd(), weights_path, 'models')): os.mkdir(os.path.join(os.getcwd(), weights_path, 'models'))
    if not os.path.exists(os.path.join(os.getcwd(), weights_path, 'loss')): os.mkdir(os.path.join(os.getcwd(), weights_path, 'loss'))
    shutil.copytree(os.path.join(os.getcwd(), 'datasets'), os.path.join(os.getcwd(), weights_path, 'datasets'), dirs_exist_ok=True)
    shutil.copytree(os.path.join(os.getcwd(), 'models'), os.path.join(os.getcwd(), weights_path, 'models'), dirs_exist_ok=True)
    shutil.copytree(os.path.join(os.getcwd(), 'loss'), os.path.join(os.getcwd(), weights_path, 'loss'), dirs_exist_ok=True)

    # get phases from dataloaders
    phases = list(dataloaders.keys())

    min_loss = 1000.0
    
    for phase in phases:
        print("Dataloder: {} set len = {}".format(phase, len(dataloaders[phase].dataset)))

    itera = None
    itera = tqdm(range(kw["train"]["epochs"]))
  
    for epoch in itera:

        for phase in phases:
            # switch mode
            if phase=="train": model.model.train()
            else: model.model.eval()

            phase_stats:List[Dict] = []

            for data, mask in dataloaders[phase]:
                # data to device
                data = tensors2device(data, device=local_rank)
                # clear grad
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    output = model(data)

                    loss, stats = loss_fn(output, mask)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                phase_stats.append(stats)
                torch.cuda.empty_cache()

                # ******* PHASE END *******
                phase_avg_stats = avg_stats(phase_stats)

                loss_fn.print_stats(epoch, phase, writer, phase_avg_stats)

        # ******* EPOCH END *******
        if scheduler is not None: 
            scheduler.step()

        if weights_path is not None and loss.item() < min_loss:
            min_loss = loss.item()
            model.save(os.path.join(weights_path, "Min_Loss.pth".format(epoch)))

if __name__ == "__main__":
    main(**parse_opt())
