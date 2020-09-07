# ANALYSIS

This repository contains the PyTorch implementation of the paper **Understanding the wiring evolution in differentiable neural architecture search**.

By Sirui Xie*, Shoukang Hu*, Xinjiang Wang, Chunxiao Liu, Jianping Shi, Xunying Liu, Dahua Lin.

[Paper-arxiv](https://arxiv.org/abs/2009.01272)

## Getting Started
* Install [PyTorch](http://pytorch.org/)
* Clone the repo:
  ```
  git clone https://github.com/SNAS-Series/SNAS-Series.git
  ```

## Requirements
* python packages
  * pytorch>=0.4.0
  * torchvision>=0.2.1
  * tensorboardX
  
* some codes are borrowed from **DARTS** ([https://github.com/quark0/darts]).

### Experiments on DSNAS
```shell
cd snas
```

Run experiments of 8 normal cells to observe patterns: **Growing tendency**  and **Width preference** (updating both network parameters and architecture parameters) 
```shell
bash train.sh dsnas 8 150 96 order 4 4 order order order order order 8NormalCell
```

Run experiments of 8 normal minimal cells to observe patterns: **Growing tendency**  and **Width preference** (updating both network parameters and architecture parameters) 
```shell
bash train.sh dsnas 8 150 96 order 2 2 order order order order order 8NormalCell
```

Calculate cost mean statistics for each operation of a minimal cell at initialization and near theta’s convergence (figure 4 in the paper)
```shell
bash train_cal_cost.sh 2020-02-19 dsnas 1 96 2 2 dsnas
```

Delete edge 0, edge 2, edge (1,3), update theta by using random sample and calaulate average cost mean statistics per epoch (figure 10b in the paper)
```shell
bash train.sh dsnas 1 150 96 order 2 1 del_edge0 del_edge2 fix_edge1_op7 random_sample random_sample del02_fix_1op7
```

Relationship of cost, loss and entropy 
```shell
bash train_sample_cost_entropy_loss.sh dsnas 1 150 96 order 2 2 fix_edge4_noskip random_sample 4_noskip_cal_cost
```

Copy plot_loss_entropy_cost.py to the experimental directory above to plot the figure (relationship of loss, entropy and cost in the appendix)
```shell
python plot_loss_entropy_cost.py 'correct loss' 150(epoch_num)
```

### Bi-level Experiments on DARTS
```shell
cd darts/cnn
```

Run experiments of 8 normal cells to observe patterns: **Catastrophic failure** (updating both network parameters and architecture parameters) 
```shell
bash train_darts.sh 8 96 4 4 8NormalCell
```

Run experiments of 8 minimal cells to observe the pattern: **Catastrophic failure** (updating both network parameters and architecture parameters) 
```shell
bash train_darts.sh 8 96 2 2 8NormalCell
```

Relationship of cost, loss and entropy 
```shell
bash train_darts_loss_entropy_cost.sh 8 50 64 2 2 fix_edge4_noskip random_sample 4_noskip_cal_cost
```

Copy plot_loss_entropy_cost.py to the experimental directory above to plot the figure (relationship of loss, entropy and cost in the appendix)

**Search set**
```shell
python plot_loss_entropy_cost.py 'search correct loss' 50 search
```

**Training set**
```shell
python plot_loss_entropy_cost.py 'train correct loss' 50 train
```

### Main results are shown above. You can also modify the code to easily reproduce other results in the paper.
