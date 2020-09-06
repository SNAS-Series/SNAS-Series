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


class DistModule(torch.nn.Module):
    def __init__(self, module, sync=False):
        super(DistModule, self).__init__()
        self.module = module
        broadcast_params(self.module)

        if not sync:
            self._grad_accs = []
            self._register_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def train(self, mode=True):
        super(DistModule, self).train(mode)
        self.module.train(mode)

    def _register_hooks(self):
        for i,(name,p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)

    def _make_hook(self, name, p, i):
        def hook(*ignore):
            link.allreduce_async(name, p.grad.data)
        return hook

def init_dist(backend='nccl',
              master_ip='127.0.0.2',
              port=29500):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

#    if '[' in node_list:
#        beg = node_list.find('[')
#        pos1 = node_list.find('-', beg)
#        if pos1 < 0:
#            pos1 = 1000
#        pos2 = node_list.find(',', beg)
#        if pos2 < 0:
#            pos2 = 1000
#        node_list = node_list[:min(pos1, pos2)].replace('[', '')
#    addr = node_list[8:].replace('-', '.')

    os.environ['MASTER_ADDR'] = str(master_ip)
    os.environ['MASTER_PORT'] = str(port)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    device = torch.device("cuda")
    dist.init_process_group(backend=backend)
    return rank, world_size, device


#def init_dist(backend='nccl', master_ip='10.10.17.21', port=29500):
#
#   if mp.get_start_method(allow_none=True) is None:
#       mp.set_start_method('spawn')
#
#   node_list = os.environ['SLURM_NODELIST']
#
#   if '[' in node_list:
#       beg = node_list.find('[')
#       pos1 = node_list.find('-', beg)
#       if pos1 < 0:
#           pos1 = 1000
#       pos2 = node_list.find(',', beg)
#       if pos2 < 0:
#           pos2 = 1000
#       node_list = node_list[:min(pos1, pos2)].replace('[', '')
#   addr = node_list[8:].replace('-', '.')
#
#   os.environ['MASTER_ADDR'] = addr
#   os.environ['MASTER_PORT'] = str(port)
#
#   rank = int(os.environ['RANK'])
#   world_size = int(os.environ['WORLD_SIZE'])
#
#   num_gpus = torch.cuda.device_count()
#   torch.cuda.set_device(rank % num_gpus)
#   device = torch.device('cuda')
#
#   dist.init_process_group(backend=backend)
#
#   return rank, world_size, device


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

#def dist_init():
#    proc_id = int(os.environ['SLURM_PROCID'])
#    ntasks = int(os.environ['SLURM_NTASKS'])
#    node_list = os.environ['SLURM_NODELIST']
#    num_gpus = torch.cuda.device_count()
#    torch.cuda.set_device(proc_id%num_gpus)
#    device = torch.device("cuda")
#
#    link.initialize()
#    world_size = link.get_world_size()
#    rank = link.get_rank()
#    
#    return rank, world_size, device



class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1):
        if world_size is None:
            world_size = link.get_world_size()
        if rank is None:
            rank = link.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter
        self.total_size = self.total_iter*self.batch_size
        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter+1)*self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

    # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(0)
        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size-1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg+self.total_size]
        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        #return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size
    
    
class CustomSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = int(os.environ['WORLD_SIZE'])
        if rank is None:
            rank = int(os.environ['RANK'])
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))
        offset = self.num_samples * self.rank
        indices = indices[offset:min(offset + self.num_samples, len(indices))]
        return iter(indices)

    def __len__(self):
        return self.num_samples

class CustomDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0))
        self.total_size = self.num_samples
        #self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        #self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))
        
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = 0
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples
        #offset = self.num_samples * self.rank
        #indices = indices[offset:offset + self.num_samples]
        #assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class CustomSplitDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, split_train_search_set = 'train', num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.dataset = dataset
        self.split_train_search_set = split_train_search_set
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if self.split_train_search_set == 'train' or self.split_train_search_set == 'search':
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0/2))
        else:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0))
        self.total_size = self.num_samples
        #self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        #self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.split_train_search_set == 'train':
            indices = list(torch.randperm(len(self.dataset)//2, generator=g))
        elif self.split_train_search_set == 'search':
            indices = list(torch.randperm(len(self.dataset)//2, generator=g)+len(self.dataset)//2)
        else:
            indices = list(torch.randperm(len(self.dataset), generator=g))        
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = 0
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples
        #offset = self.num_samples * self.rank
        #indices = indices[offset:offset + self.num_samples]
        #assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
