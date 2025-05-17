import os, sys
import torch
import argparse
import numpy as np
import yaml
from evaluate.utils import get_embeddings, get_hetero_recall_precision
from misc.utils import get_datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import importlib
def import_module_from_path(module_name, module_path):
    module = importlib.import_module(module_path)
    return getattr(module, module_name)
x_label_font = 20
y_label_font = 20
legend_font = 20
fig_size_x = 6.5            
fig_size_y = 6.5
def parse_opt()->dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml",    type=str, default='config/evaluate/test.yaml', required=False)
    parser.add_argument("--tn",      type=int, default=1) 
    parser.add_argument("--rp",      type=int, default=100)
    parser.add_argument("--save",    type=str, default="results/evaluate/")

    parser.add_argument("--pre_calculate_distance",type=str, default=None)
    
    # parser.add_argument("--pre_calculate_distance",type=list, default = [
    #     '/results/evaluate/TIME',
    # ])
    
    parser.add_argument("--pre_calculate_distance_model_name",type=list, default=[
        'HBFusion',
    ])

    opt = parser.parse_args()
    opt = vars(opt)
    f = open(opt["yaml"], encoding="utf-8")
    eval = yaml.load(f, Loader=yaml.FullLoader)
    
    eval.update(opt)
    return eval

def average_PR_curve(geneous, sp, all_PR_curve, sample_proportion):
    if geneous == 'aerial-aerial' or geneous == 'ground-ground':
        if geneous == 'ground-ground':
            duple_geneous = 'aerial-aerial'
        else:
            duple_geneous = 'ground-ground'
        all_sample_propotion = sample_proportion
        '''get key'''
        keys = [f'{sp}_{j}' for j in all_sample_propotion]
        select_geneous = []
        for k in keys:
            k_ = k
            if float(k.split('_')[0]) < float(k.split('_')[1]):
                k_ = '_'.join([k.split('_')[1], k.split('_')[0]])
            select_geneous.append(f'{k_}_{geneous}')
            select_geneous.append(f'{k_}_{duple_geneous}')
        select_PR_curve = []
        for s in select_geneous:
            select_PR_curve.append(all_PR_curve[s]['xy'])
        select_PR_curve = [np.average(select_PR_curve, axis=0)]
        pr_curve_jingdu = 50
        x = np.linspace(0, 1, pr_curve_jingdu)
        y = []
    
        for s in select_PR_curve:
            y_ = np.linspace(0, 1, pr_curve_jingdu)
            for i, x_ in enumerate(x):
                position = s[0].searchsorted(x_)
                y_[i] = s[1][position-1] + (x_ - s[0][position-1]) * (s[1][position ] - s[1][position-1]) / (s[0][position] - s[0][position-1])

            y.append(y_)
        return np.concatenate([x.reshape(1, -1), np.clip(np.mean(np.array(y), axis=0).reshape(1, -1), 0, 1)], axis=0)
 
    else:
        all_sample_propotion = sample_proportion
        '''get key'''
        keys = [f'{sp}_{j}' for j in all_sample_propotion]
        select_geneous = []
        for k in keys:
            k_ = k
            if float(k.split('_')[0]) < float(k.split('_')[1]):
                k_ = '_'.join([k.split('_')[1], k.split('_')[0]])
            select_geneous.append(f'{k_}_{geneous}')
        select_PR_curve = []
        for s in select_geneous:
            select_PR_curve.append(all_PR_curve[s]['xy'])
            
        pr_curve_jingdu = 50
        x = np.linspace(0, 1, pr_curve_jingdu)
        y = []
    
        for s in select_PR_curve:
            y_ = np.linspace(0, 1, pr_curve_jingdu)
            for i, x_ in enumerate(x):
                position = s[0].searchsorted(x_)
                y_[i] = s[1][position-1] + (x_ - s[0][position-1]) * (s[1][position ] - s[1][position-1]) / (s[0][position] - s[0][position-1])
            y.append(y_)
        return np.concatenate([x.reshape(1, -1), np.clip(np.mean(np.array(y), axis=0).reshape(1, -1), 0, 1)], axis=0)

def feat_l2d_mat(embeddings_s: np.ndarray, embeddings_q: np.ndarray) -> np.ndarray:
    Nm, Fs = embeddings_s.shape
    distance = np.linalg.norm(embeddings_s.reshape((Nm, 1, Fs)) - embeddings_q.reshape((1, Nm, Fs)), axis=2)
    distance += np.eye(Nm)*(np.max(distance)+1) 
    return distance

def draw_PR_curve(geneous_list, sample_proportion, name_list, savepath, all_PR_curve):
    colors = ['b', 'g', 'r', 'c', 'm']
    for g in geneous_list:
        for j in sample_proportion:
            plt.figure()
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)
            plt.xticks(fontsize = x_label_font)
            plt.yticks(fontsize = y_label_font)
            plt.grid()
            for i_, n in enumerate(name_list): 
                ave_cur = average_PR_curve(g, j, all_PR_curve[n], sample_proportion)
                plt.plot(ave_cur[0], ave_cur[1], color=colors[i_],)

            plt.legend(name_list, prop={'size': legend_font}, loc='upper right')
            fig = plt.gcf()
            fig.set_size_inches(fig_size_x, fig_size_y)
            if os.path.join(savepath, g) is not None: plt.savefig(os.path.join(savepath, g, f"PR_curve_{g}_{j*100}%.png"), dpi=600)
            plt.close()

    for g in geneous_list:
        plt.figure()
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xticks(fontsize = x_label_font)
        plt.yticks(fontsize = y_label_font)
        plt.grid()
        for i_, n in enumerate(name_list):
            ave_cur = []
            for j in sample_proportion:
                ave_cur.append(average_PR_curve(g, j, all_PR_curve[n], sample_proportion))
            ave_cur = np.array(ave_cur)
            plt.fill_between(np.mean(ave_cur, axis=0)[0], np.min(ave_cur, axis=0)[1], np.max(ave_cur, axis=0)[1], alpha=0.2, linewidth=0.1, color=colors[i_])

        plt.legend(name_list, prop={'size': legend_font}, loc='lower left')
        for i_, n in enumerate(name_list):
            ave_cur = []
            for j in sample_proportion:
                ave_cur.append(average_PR_curve(g, j, all_PR_curve[n], sample_proportion))
            ave_cur = np.array(ave_cur)
            plt.plot(np.mean(ave_cur, axis=0)[0], np.mean(ave_cur, axis=0)[1], color=colors[i_])
            
        fig = plt.gcf()
        fig.set_size_inches(fig_size_x, fig_size_y)
        if os.path.join(savepath, g) is not None: plt.savefig(os.path.join(savepath, g, f"PR_curve_{g}.png"), dpi=600)
        plt.close()

    
    print('Save PR_curve.png')
    return

def main(**kw):
    savepath = None

    sample_proportion = [1.0, 0.7, 0.5, 0.35, 0.25][:1]
    
    if kw['pre_calculate_distance'] is None:
        weights_name = kw['weights'].split('/')[-2]
        sys.path.insert(0, f'pretrain/{weights_name}')

        print(f'LOADING model weight {weights_name}')

        GroundAerialDataLoader = import_module_from_path('GroundAerialDataLoader', f'pretrain.{weights_name}.datasets.dataloders.GroundAerialdataloader')
        Model = import_module_from_path('Model', f'pretrain.{weights_name}.models.model')

        dataloader = GroundAerialDataLoader(**kw["dataloaders"]["evaluate"])

        device:str = None
        if torch.cuda.is_available(): device = "cuda"
        else: device = "cpu"
        print("Device: {}".format(device))
        assert os.path.exists(kw["weights"]), "Cannot open network weights: {}".format(kw["weights"])
        print("Loading weights: {}".format(kw["weights"]))

        model = Model()
        model.load(kw["weights"], device)

        ''' save different sample proportion qembedding '''
        embedding_dict = {}
        for i in sample_proportion:
            print(f'point cloud sample proportion is {i}')
            kw['dataloaders']['evaluate']['dataset']['sample_proportion'] = i
            embedding_dict[i] = get_embeddings(model, dataloader, device, print_stats=True)
        
        if kw["save"] is not None:
            assert os.path.exists(kw["save"]), "Path does not exist, please run: mkdir " + kw["save"]
            
        savepath = os.path.join(kw["save"], get_datetime())
        os.mkdir(savepath)
        print("Save path:", savepath)
        if savepath is not None: np.save(os.path.join(savepath, "embedding_dict.npy"), embedding_dict)
        
        ''' save feature distances of different sample proportion '''
        distanceS_results = {}
        for i_, i in enumerate(sample_proportion):
            for j in sample_proportion[i_:]:
                print(f'Calculate PR curve at {i} proportion for source and {j} proportion for query')
                distanceS_results[f'{i}_{j}'] = feat_l2d_mat(embedding_dict[i], embedding_dict[j])
        if savepath is not None: np.save(os.path.join(savepath, "distances_all.npy"), distanceS_results)

        ''' save PR_curve results '''
        distances_all = {}
        
        for i in distanceS_results:
            rp = get_hetero_recall_precision(dataloader.dataset, distanceS_results[i], num_eval=kw["rp"])
            for r in rp:
                distances_all[f'{i}_{r}'] = rp[r]
        if savepath is not None: np.save(os.path.join(savepath, "PR_results_all.npy"), distances_all)
        
        return
    
    else:
        all_PR_curve = {}
        for i_, i in enumerate(kw['pre_calculate_distance_model_name']):
            all_PR_curve[i] = np.load(os.path.join(kw['pre_calculate_distance'][i_], 'PR_results_all.npy'), allow_pickle=True).item()
        
        savepath = kw['pre_calculate_distance'][i_]
        print(f'New save path is {savepath}')
        
    geneous_list = [i.split('_')[-1] for i in all_PR_curve[i]]
    geneous_list = list(set(geneous_list))
    if 'all-all' in geneous_list:
        geneous_list.remove('all-all')
    proportion_list = ['_'.join(i.split('_')[:2]) for i in all_PR_curve[i]]
    proportion_list = list(set(proportion_list))
    proportion_list.sort(reverse=True)
    name_list = kw['pre_calculate_distance_model_name']

    for g in geneous_list:
        if not os.path.exists(os.path.join(savepath, g)): os.mkdir(os.path.join(savepath, g))
    
    ''' draw PR Curve '''
    draw_PR_curve(geneous_list, sample_proportion, name_list, savepath, all_PR_curve)

    return


if __name__ == "__main__":
    main(**parse_opt())