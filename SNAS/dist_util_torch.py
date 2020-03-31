import os
import time
import math
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler
import multiprocessing as mp
import numpy as np
from torch.distributed import get_world_size, get_rank

# def init_dist(backend='nccl',
#               master_ip='127.0.0.2',
#               port=29500):
#     if mp.get_start_method(allow_none=True) is None:
#         mp.set_start_method('spawn')

# #    if '[' in node_list:
# #        beg = node_list.find('[')
# #        pos1 = node_list.find('-', beg)
# #        if pos1 < 0:
# #            pos1 = 1000
# #        pos2 = node_list.find(',', beg)
# #        if pos2 < 0:
# #            pos2 = 1000
# #        node_list = node_list[:min(pos1, pos2)].replace('[', '')
# #    addr = node_list[8:].replace('-', '.')

#     os.environ['MASTER_ADDR'] = str(master_ip)
#     os.environ['MASTER_PORT'] = str(port)
#     rank = int(os.environ['RANK'])
#     world_size = int(os.environ['WORLD_SIZE'])
#     num_gpus = torch.cuda.device_count()
#     torch.cuda.set_device(rank % num_gpus)
#     device = torch.device("cuda")
#     dist.init_process_group(backend=backend)
#     return rank, world_size, device


def init_dist(backend='nccl', master_ip='10.10.17.21', port=29500):

   if mp.get_start_method(allow_none=True) is None:
       mp.set_start_method('spawn')

   # node_list = os.environ['SLURM_NODELIST']

   # if '[' in node_list:
   #     beg = node_list.find('[')
   #     pos1 = node_list.find('-', beg)
   #     if pos1 < 0:
   #         pos1 = 1000
   #     pos2 = node_list.find(',', beg)
   #     if pos2 < 0:
   #         pos2 = 1000
   #     node_list = node_list[:min(pos1, pos2)].replace('[', '')
   # addr = node_list[8:].replace('-', '.')

   # os.environ['MASTER_ADDR'] = addr
   os.environ['MASTER_ADDR'] = master_ip
   os.environ['MASTER_PORT'] = str(port)

   rank = int(os.environ['RANK'])
   world_size = int(os.environ['WORLD_SIZE'])

   num_gpus = torch.cuda.device_count()
   torch.cuda.set_device(rank % num_gpus)
   device = torch.device('cuda')

   dist.init_process_group(backend=backend)

   return rank, world_size, device


def reduce_gradients(model, sync=False):
    """ average gradients """
    for name, param in model.named_parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.detach())

def reduce_tensorgradients(tensor_list, sync=False):
    """ average gradients """
    for param in tensor_list:
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad.detach())

def part_reduce_gradients(tensor_list, param_count, sync=False):
    """ average gradients """
    dist.all_reduce(param_count.detach())
    id = 0
    for param in tensor_list:
        if param.requires_grad:
            if param_count[id]!= 0:
                dist.all_reduce(param.grad.data)                    
                param.grad.div_(param_count[id])
            id += 1

def broadcast_params(model):
    """ broadcast model parameters """
    for name,p in model.state_dict().items():
        dist.broadcast(p, 0)


