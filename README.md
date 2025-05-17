# HBFusion

## Introduction
[T-ITS 25] Cross-Source Place Recognition for Unmanned Aerial-Ground Vehicles with Low-Overlap and Varying-Density Point Cloud

This article proposes a hierarchical BEV fusion network for unmanned aerial-ground vehicles with low-overlap and varying-density point clouds, called HBFusion.

<div align="center">
    <div align="center">
        <img src="media/pipeline.png" width = 90% >
    </div>
    <font>
    HBFusion Pipeline</font>
</div>

## Anaconda Environment Setup

Please use the following command for installation.

```bash
conda create -n hbfusion python==3.8.20
conda activate hbfusion
pip install torch==1.9.1+cu111
pip install pip install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

## Dataset
We put self-recorded datasets used in HBFusion in [Quark](https://pan.quark.cn/s/9c012d92bcd1). 

## Train and Test

1. Change the path of dataset in `config/train/train.yaml`.

2. train model
```
python train.py
```
When you finish training model, you can get weights in weights/ (containing DATASETS, MODELS, LOSS). Copy weights/ to pretrain/ and rename it as you like. 

3. Test pipeline1 (fixed point density): Run test.py directly and obtain distances_all.npy, embedding_dict.npy, and PR_results_all.npy in path results/evaluate/{Time}. Comment out the code on line 28 in test.py and uncomment the code on lines 30-32. Then you MUST change 'pre_calculate_distance' on lines 30-32 to results/evaluate/{Time} and run test.py again. Finally you can obatin experimental results.

4. Test pipeline2 (varying point densities): Same as Test pipeline1 but change (line 165 in test.py) sample_proportion = [1.0, 0.7, 0.5, 0.35, 0.25][:1] to sample_proportion = [1.0, 0.7, 0.5, 0.35, 0.25], before you run test.py.

# Acknowledgement
We acknowledge the authors of [GAPR](https://github.com/SYSU-RoboticsLab/GAPR) and [TorchSparse](https://github.com/mit-han-lab/torchsparse) for their excellent codebase which has been used as a starting point for this project.

