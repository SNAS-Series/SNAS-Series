import pdb
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
from torch.utils import checkpoint as cp
import csv
import sys
import os
from utils import drop_path
from ast import literal_eval

import torch.distributed as dist

from model_edge_all import NetworkChild, AuxiliaryHeadCIFAR
from dist_util_torch import init_dist, broadcast_params, reduce_gradients, reduce_tensorgradients, part_reduce_gradients

class MixedOp(nn.Module):
    def __init__(self, C, stride, op_size, op_flops, op_mac, primitives, bn_affine):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._resource_size = op_size
        self._resource_flops = op_flops
        self._resource_mac = op_mac
        self.got_flops_mac = False
        self.Primitives = primitives
        for primitive in self.Primitives:
            op = OPS[primitive](C, stride, bn_affine)  # make BatchNorm2 update while search
            self._resource_size[self.Primitives.index(primitive)] = op.size
            self._ops.append(op)

    def forward(self, x, weights):
        if self.got_flops_mac:
            result = sum(w * op(x) for w, op in zip(weights, self._ops))
        else:
            result = 0
            index = 0
            for w, op in zip(weights, self._ops):
                result += w * op(x)
                self._resource_flops[index] = op.flops
                self._resource_mac[index] = op.mac
                index += 1
            self.got_flops_mac = True
        return result


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, primitives, bn_affine, use_ckpt=True):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.use_ckpt = use_ckpt
        self.Primitives = primitives
        self.bn_affine = bn_affine
        self.device = torch.device("cuda")

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=self.bn_affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=self.bn_affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=self.bn_affine)
        self._steps = steps
        self._multiplier = multiplier

        self._k = sum(1 for i in range(self._steps) for n in range(2 + i))
        self._num_ops = len(self.Primitives)
        self.op_size = (torch.zeros(self._k, self._num_ops)).to(self.device)
        self.op_flops = (torch.zeros(self._k, self._num_ops)).to(self.device)
        self.op_mac = (torch.zeros(self._k, self._num_ops)).to(self.device)

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        count = 0
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, self.op_size[count], self.op_flops[count], self.op_mac[count], self.Primitives,
                             self.bn_affine)
                self._ops.append(op)
                count += 1

    def forward(self, s0, s1, weights, drop_path_prob = 0):
        if self.use_ckpt:
            s0 = cp.checkpoint(self.preprocess0, s0)
            s1 = cp.checkpoint(self.preprocess1, s1)
        else:
            s0 = self.preprocess0(s0)
            s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = 0
            for j, h in enumerate(states):
                op = self._ops[offset + j]
                if self.use_ckpt:
                    h = cp.checkpoint(op, *[h, weights[offset + j]])
                else:
                    h = op(h, weights[offset + j])
                if self.training and drop_path_prob > 0.:
                    if not isinstance(op, Identity):
                        h = drop_path(h, drop_path_prob)
                s += h
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1), self.op_size, self.op_flops, self.op_mac


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, args, rank, world_size,
                 steps=4, multiplier=4, stem_multiplier=3): #, use_ckpt=True):
        super(Network, self).__init__()
        self.device = torch.device("cuda")
        self.snas = args.snas
        self.dsnas = args.dsnas
        self._world_size = world_size
        self._use_ckpt = args.use_ckpt
        self._resample_layer = args.resample_layer
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._resource_efficient = args.resource_efficient
        self._resource_lambda = args.resource_lambda
        self._method = args.method
        self._drop_path_prob = args.drop_path_prob
        #self._drop_path_prob = 0
        # self._discrete = args.discrete
        self._normalization = args.normalization
        self._running_mean_var = args.running_mean_var
        self._separation = args.separation
        self._log_penalty = args.log_penalty
        self._loss = args.loss
        if args.ckpt_false_list != 'all':
            self._ckpt_false_list = literal_eval(args.ckpt_false_list)
        else:
            self._ckpt_false_list = range(self._layers)
        self._bn_affine = args.bn_affine
        self._distributed = args.distributed
        self._minus_baseline = args.minus_baseline
        self._loc_mean = args.loc_mean
        self._loc_std = args.loc_std
        self._temp = args.temp
        self._nsample = args.nsample
        self._steps = steps
        self._multiplier = multiplier
        self._stem_multiplier = stem_multiplier
        self._init_channels = args.init_channels
        self._auxiliary = args.auxiliary
        self.args = args
        self._k = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.Primitives = PRIMITIVES
        self._num_ops = len(self.Primitives)

        if self._distributed:
            self.normal_log_alpha = torch.nn.Parameter(
                torch.zeros(self._k, self._num_ops).normal_(self._loc_mean,
                                                                          self._loc_std).cuda().requires_grad_())
            self.reduce_log_alpha = torch.nn.Parameter(
                torch.zeros(self._k, self._num_ops).normal_(self._loc_mean,
                                                            self._loc_std).cuda().requires_grad_())
        else:
            self.normal_log_alpha = Variable(
                torch.zeros(self._k, self._num_ops).normal_(self._loc_mean, self._loc_std).cuda(),
                requires_grad=True)
            self.reduce_log_alpha = Variable(
                torch.zeros(self._k, self._num_ops).normal_(self._loc_mean, self._loc_std).cuda(),
                requires_grad=True)

            self.normal_log_alpha_ema = Variable(
                torch.zeros(self._k, self._num_ops).normal_(self._loc_mean, self._loc_std).cuda(),
                requires_grad=True)
            self.reduce_log_alpha_ema = Variable(
                torch.zeros(self._k, self._num_ops).normal_(self._loc_mean, self._loc_std).cuda(),
                requires_grad=True)

        self.normal_edge_reward = torch.zeros(self.normal_log_alpha.size(0)).to(self.device)
        self.reduce_edge_reward = torch.zeros(self.reduce_log_alpha.size(0)).to(self.device)
        self.normal_log_alpha_pre = self.normal_log_alpha.clone().detach()
        self.reduce_log_alpha_pre = self.reduce_log_alpha.clone().detach()

        self.normal_edge_reward_running_mean = torch.zeros(1).to(self.device)
        self.normal_edge_reward_running_var = torch.zeros(1).to(self.device)
        self.reduce_edge_reward_running_mean = torch.zeros(1).to(self.device)
        self.reduce_edge_reward_running_var = torch.zeros(1).to(self.device)

        self._arch_parameters = [
            self.normal_log_alpha,
            self.reduce_log_alpha
        ]
        self._rank = rank
        self._logger = None
        self._logging = None

        self.net_init()

    def net_init(self):
        C = self._C
        layers = self._layers
        steps = self._steps
        multiplier = self._multiplier
        stem_multiplier = self._stem_multiplier
        num_classes = self._num_classes
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        self.reduce_list = [layers // 3, 2 * layers // 3]
        self.num_reduce = len(self.reduce_list)
        self.num_normal = layers - self.num_reduce
        for i in range(layers):
            if i in self.reduce_list:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            if self._use_ckpt:
                if i in self._ckpt_false_list:
                    cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                                self.Primitives, self._bn_affine, use_ckpt=False)
                else:
                    cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                                self.Primitives, self._bn_affine,  use_ckpt=True)
            else:
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                            self.Primitives, self._bn_affine, use_ckpt=False)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        
        if self._auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def logp(self, log_alpha, weights):
        # log_alpha 2d weights 2d
        lam = self._num_ops
        temp = self._temp
        epsilon = 1e-4
        epsilon_weight = torch.ones_like(weights) * epsilon
        weights_temp = torch.max(weights, epsilon_weight).to(self.device)
        last_term_epsilon = torch.max((weights_temp ** (-temp)) * torch.exp(log_alpha), epsilon_weight).to(self.device)
        log_prob = math.log(5040) + (lam - 1) * math.log(temp) + log_alpha.sum(-1) - \
                   (temp + 1) * (torch.log(weights_temp).sum(-1)) - lam * torch.log(last_term_epsilon.sum(-1))
        return log_prob


    def forward(self, input, target=None, criterion=None, input_search=None, target_search=None):

        total_penalty = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Dropout penalty, entropy penalty, policy gradient,
        # resource size_expectation, dropout sparse, resource_flop_E, resource_mac_E
        # reduce_policy_gradient, reduce_size_exp, reduce_flop_exp, reduce_mac_exp
        # 2 - total_penalty of policy_gradient for normal cell
        # 3 - expectation of size for normal cell

        # 25 -
        op_normal = [0, 0, 0]
        op_reduce = [0, 0, 0]
        logits_aux = None

        if not self._resample_layer:
            normal_weights = self._get_weights(self.normal_log_alpha)
            reduce_weights = self._get_weights(self.reduce_log_alpha)
            self.normal_weights = normal_weights
            self.reduce_weights = reduce_weights

        if self.args.dsnas:
            if self.args.gen_max_child_flag and not self.training:
                normal_weights = torch.zeros_like(self.normal_log_alpha).scatter_(1, torch.argmax(self.normal_log_alpha, dim = -1).view(-1,1), 1)
                reduce_weights = torch.zeros_like(self.reduce_log_alpha).scatter_(1, torch.argmax(self.reduce_log_alpha, dim = -1).view(-1,1), 1)

            normal_one_hot_prob = (normal_weights * F.softmax(self.normal_log_alpha, dim=-1)).sum(-1)
            reduce_one_hot_prob = (reduce_weights * F.softmax(self.reduce_log_alpha, dim=-1)).sum(-1)
            
            genotype_child = self.genotype_child(normal_weights, reduce_weights)
            model_child = NetworkChild(self._init_channels, self._num_classes, self._layers, 
    self._auxiliary, genotype_child, self.Primitives, self._drop_path_prob, self._use_ckpt, self._bn_affine)
            model_child = model_child.to(self.device)
            if not self.training:
                model_child.eval()
#            model_child.train()
            
            self.load_child_state_dict(model_child)
            normal_weights.requires_grad_()
            reduce_weights.requires_grad_()
            logits, logits_aux = model_child(input, normal_weights, reduce_weights)
            
            error_loss = criterion(logits, target)
#            loss_alpha = (normal_one_hot_prob + reduce_one_hot_prob).sum()
            if self.args.prox_policy_opt:  
                normal_one_hot_prob_pre = (normal_weights.detach() * F.softmax(self.normal_log_alpha_pre, dim=-1)).sum(-1)
                reduce_one_hot_prob_pre = (reduce_weights.detach() * F.softmax(self.reduce_log_alpha_pre, dim=-1)).sum(-1)
                normal_ratio = normal_one_hot_prob/normal_one_hot_prob_pre
                reduce_ratio = reduce_one_hot_prob/reduce_one_hot_prob_pre
                normal_ratio_clip = torch.clamp(normal_ratio, min=1-self.args.prox_policy_epi, max=1+self.args.prox_policy_epi)
                reduce_ratio_clip = torch.clamp(reduce_ratio, min=1-self.args.prox_policy_epi, max=1+self.args.prox_policy_epi)
                loss_alpha = (torch.log(torch.max(normal_ratio, normal_ratio_clip))+torch.log(torch.max(reduce_ratio, reduce_ratio_clip))).sum()
            else:
                loss_alpha = (torch.log(normal_one_hot_prob) + torch.log(reduce_one_hot_prob)).sum()
            
            if self.args.auxiliary and self.training:
                loss_aux = criterion(logits_aux, target)
                error_loss += self.args.auxiliary_weight*loss_aux

            if self.training:
                
                if self.args.add_entropy_loss:
                    entropy_loss = self._arch_entropy(self.normal_log_alpha) + self._arch_entropy(self.reduce_log_alpha)
                    loss = error_loss.clone() + loss_alpha.clone() + entropy_loss
                else:    
                    loss = error_loss.clone() + loss_alpha.clone()
                if self.args.distributed:
                    loss.div_(self._world_size)

                for v in model_child.parameters():
                    if v.grad is not None:
                        v.grad = None
                loss.backward()

                if self.args.edge_reward_norm:
                    normal_edge_reward_mean = normal_weights.grad.sum(-1).mean()
                    normal_edge_reward_var = normal_weights.grad.sum(-1).std()
                    reduce_edge_reward_mean = reduce_weights.grad.sum(-1).mean()
                    reduce_edge_reward_var = reduce_weights.grad.sum(-1).std()
                    
                    self.normal_edge_reward_running_mean =  self.normal_edge_reward_running_mean * 0.1 + normal_edge_reward_mean * 0.9
                    self.normal_edge_reward_running_var =  self.normal_edge_reward_running_var * 0.1 + normal_edge_reward_var * 0.9
                    self.reduce_edge_reward_running_mean =  self.reduce_edge_reward_running_mean * 0.1 + reduce_edge_reward_mean * 0.9
                    self.reduce_edge_reward_running_var =  self.reduce_edge_reward_running_var * 0.1 + reduce_edge_reward_var * 0.9
                    
                    
                    if not self.args.edge_reward_norm_mean_0:
                        self.normal_edge_reward = (normal_weights.grad.sum(-1)-self.normal_edge_reward_running_mean)/self.normal_edge_reward_running_var
                        self.reduce_edge_reward = (reduce_weights.grad.sum(-1)-self.reduce_edge_reward_running_mean)/self.reduce_edge_reward_running_var
                    else:
                        self.normal_edge_reward = normal_weights.grad.sum(-1)/self.normal_edge_reward_running_var
                        self.reduce_edge_reward = reduce_weights.grad.sum(-1)/self.reduce_edge_reward_running_var 
 
                else:
                    self.normal_edge_reward = normal_weights.grad.sum(-1)
                    self.reduce_edge_reward = reduce_weights.grad.sum(-1)
                    
                
                self.normal_log_alpha.grad = self.normal_log_alpha.grad.detach()*self.normal_edge_reward.view(-1,1)
                self.reduce_log_alpha.grad = self.reduce_log_alpha.grad.detach()*self.reduce_edge_reward.view(-1,1)

                state_dict = self.state_dict()
                child_state_dict = model_child.state_dict()
               
#                self.load_state_dict(child_state_dict, strict=False) 
                for model_child_name, model_child_param in model_child.named_parameters():
                    if model_child_param.grad is not None:
                        state_dict[model_child_name].grad = model_child_param.grad.clone().detach()

                for model_name, model_param in self.named_parameters():
                    if state_dict[model_name].grad is not None:
                        model_param.grad = state_dict[model_name].grad.clone().detach()       
                
                if self.args.prox_policy_opt:  
                    self.normal_log_alpha_pre = self.normal_log_alpha.clone().detach()
                    self.reduce_log_alpha_pre = self.reduce_log_alpha.clone().detach()

        if self.snas or self.training:
            
            if self.args.gen_max_child_flag and not self.training:
                normal_weights = torch.zeros_like(self.normal_log_alpha).scatter_(1, torch.argmax(self.normal_log_alpha, dim = -1).view(-1,1), 1)
                reduce_weights = torch.zeros_like(self.reduce_log_alpha).scatter_(1, torch.argmax(self.reduce_log_alpha, dim = -1).view(-1,1), 1)
            
            s0 = s1 = self.stem(input)
            for i, cell in enumerate(self.cells):
                if self._resample_layer:
                    if cell.reduction:
                        log_alpha = self.reduce_log_alpha
                    else:
                        log_alpha = self.normal_log_alpha
                    weights = self._get_weights(log_alpha)
                else:
                    if cell.reduction:
                        log_alpha = self.reduce_log_alpha
                        weights = reduce_weights
                    else:
                        log_alpha = self.normal_log_alpha
                        weights = normal_weights
                
                s0, result = s1, cell(s0, s1, weights, self._drop_path_prob)
                s1 = result[0]
                
                if i == 2 * self._layers // 3:
                    if self._auxiliary and self.training:
                        logits_aux = self.auxiliary_head(s1)

                op_size = result[1]
                op_flops = result[2]
                op_mac = result[3]
                #if self._resource_efficient:
                discrete_prob_1 = F.softmax(log_alpha, dim=-1)

                resource_size_baseline = op_size * discrete_prob_1 # dimension: edge_num * op_num
                resource_flops_baseline = op_flops * discrete_prob_1
                resource_mac_baseline = op_mac * discrete_prob_1

                clean_size_baseline = (resource_size_baseline.sum(-1)).clone() # dimension: edge_num * 1
                clean_size_baseline[torch.abs(resource_size_baseline.sum(-1)) < 1] = 1
                clean_flops_baseline = (resource_flops_baseline.sum(-1)).clone()
                clean_flops_baseline[torch.abs(resource_flops_baseline.sum(-1)) < 1] = 1
                clean_mac_baseline = (resource_mac_baseline.sum(-1)).clone()
                clean_mac_baseline[torch.abs(resource_mac_baseline.sum(-1)) < 1] = 1

                log_resource_size_baseline = torch.log(clean_size_baseline)
                log_resource_flops_baseline = torch.log(clean_flops_baseline)
                log_resource_mac_baseline = torch.log(clean_mac_baseline)

                resource_size_average = torch.tensor(np.average((op_size.sum(0) / op_size.shape[0]).tolist()).item(), device=op_size.device)
                resource_flops_average = torch.tensor(np.average((op_flops.sum(0) / op_flops.shape[0]).tolist()).item(), device=op_flops.device)
                resource_mac_average = torch.tensor(np.average((op_mac.sum(0) / op_mac.shape[0]).tolist()).item(), device=op_mac.device)

                clean_size_average = (resource_size_average.sum(-1)).clone()  # dimension: edge_num * 1
                clean_size_average[torch.abs(resource_size_average.sum(-1)) < 1] = 1
                clean_flops_average = (resource_flops_average.sum(-1)).clone()
                clean_flops_average[torch.abs(resource_flops_average.sum(-1)) < 1] = 1
                clean_mac_average = (resource_mac_average.sum(-1)).clone()
                clean_mac_average[torch.abs(resource_mac_average.sum(-1)) < 1] = 1

                log_resource_size_average = torch.log(clean_size_average)
                log_resource_flops_average = torch.log(clean_flops_average)
                log_resource_mac_average = torch.log(clean_mac_average)

                resource_size = op_size * weights # dimension: edge_num * op_num
                resource_flops = op_flops * weights
                resource_mac = op_mac * weights

                clean_size = (resource_size.sum(-1)).clone() # dimension: edge_num * 1
                clean_flops = (resource_flops.sum(-1)).clone()
                clean_mac = (resource_mac.sum(-1)).clone()
                clean_size[torch.abs(resource_size.sum(-1)) < 1] = 1
                clean_flops[torch.abs(resource_flops.sum(-1)) < 1] = 1
                clean_mac[torch.abs(resource_mac.sum(-1)) < 1] = 1

                log_resource_size = torch.log(torch.abs(clean_size)) # dimension: edge_num * 1
                log_resource_flops = torch.log(torch.abs(clean_flops))
                log_resource_mac = torch.log(torch.abs(clean_mac))
                
                resource_size_minus_average = resource_size.sum(-1) - resource_size_average
                resource_flops_minus_average = resource_flops.sum(-1) - resource_flops_average
                resource_mac_minus_average = resource_mac.sum(-1) - resource_mac_average

                log_resource_size_minus_average = log_resource_size - log_resource_size_average
                log_resource_flops_minus_average = log_resource_flops - log_resource_flops_average
                log_resource_mac_minus_average = log_resource_mac - log_resource_mac_average

                if self._method == 'reparametrization':
                    if self._separation == 'all':
                        resource_penalty = ((resource_size * 2 + resource_flops / 4000 + resource_mac / 100) * 0.43).sum(-1)  # dimension: edge_num * 1
                        log_resource_penalty = (log_resource_size + log_resource_flops + log_resource_mac) / 3
                    elif self._separation == 'size':
                        resource_penalty = resource_size.sum(-1)
                        log_resource_penalty = log_resource_size
                    elif self._separation == 'flops':
                        resource_penalty = resource_flops.sum(-1)
                        log_resource_penalty = log_resource_flops
                    elif self._separation == 'mac':
                        resource_penalty = resource_mac.sum(-1)
                        log_resource_penalty = log_resource_mac
                    else:
                        resource_penalty = torch.zeros_like(resource_size.sum(-1))
                        log_resource_penalty = resource_penalty

                elif self._method == 'policy_gradient':
                    if self._separation == 'all':
                        if self._minus_baseline:
                            resource_penalty = (resource_size_minus_average * 2 + resource_flops_minus_average / 4000 + resource_mac_minus_average / 100) * 0.43
                            log_resource_penalty = (log_resource_size_minus_average + log_resource_flops_minus_average + log_resource_mac_minus_average) / 3
                        else:
                            resource_penalty = ((resource_size * 2 + resource_flops / 4000 + resource_mac / 100) * 0.43).sum(-1)
                            log_resource_penalty = (log_resource_size + log_resource_flops + log_resource_mac) / 3
                    elif self._separation == 'size':
                        if self._minus_baseline:
                            resource_penalty = resource_size_minus_average
                            log_resource_penalty = log_resource_size_minus_average
                        else:
                            resource_penalty = resource_size.sum(-1)
                            log_resource_penalty = log_resource_size
                    elif self._separation == 'flops':
                        if self._minus_baseline:
                            resource_penalty = resource_flops_minus_average
                            log_resource_penalty = log_resource_flops_minus_average
                        else:
                            resource_penalty = resource_flops.sum(-1)
                            log_resource_penalty = log_resource_flops
                    elif self._separation == 'mac':
                        if self._minus_baseline:
                            resource_penalty = resource_mac_minus_average
                            log_resource_penalty = log_resource_mac_minus_average
                        else:
                            resource_penalty = resource_mac.sum(-1)
                            log_resource_penalty = log_resource_mac
                    else:
                        resource_penalty = torch.zeros_like(resource_size.sum(-1))
                        log_resource_penalty = resource_penalty

                elif self._method == 'discrete':
                    if self._separation == 'all':
                        resource_penalty = ((resource_size_baseline * 2 + resource_flops_baseline / 4000 +
                                             resource_mac_baseline / 100) * 0.43).sum(-1)
                        log_resource_penalty = (log_resource_size_baseline + log_resource_flops_baseline +
                                                log_resource_mac_baseline) / 3
                    elif self._separation == 'size':
                        resource_penalty = resource_size_baseline.sum(-1)
                        log_resource_penalty = log_resource_size_baseline
                    elif self._separation == 'flops':
                        resource_penalty = resource_flops_baseline.sum(-1)
                        log_resource_penalty = log_resource_flops_baseline
                    elif self._separation == 'mac':
                        resource_penalty = resource_mac_baseline.sum(-1)
                        log_resource_penalty = log_resource_mac_baseline
                    else:
                        resource_penalty = torch.zeros_like(resource_size_baseline.sum(-1))
                        log_resource_penalty = resource_penalty
                else:
                    resource_penalty = torch.zeros_like(resource_size_baseline.sum(-1))
                    log_resource_penalty = resource_penalty

                if self._method == 'policy_gradient':
                    concrete_log_prob = self.logp(log_alpha, weights)

                    # make resource non-differentiable
                    resource_penalty = resource_penalty.data
                    log_resource_penalty = log_resource_penalty.data

                    if cell.reduction:
                        total_penalty[7] += (concrete_log_prob * resource_penalty).sum()
                        total_penalty[36] += (concrete_log_prob * log_resource_penalty).sum()
                    else:
                        total_penalty[2] += (concrete_log_prob * resource_penalty).sum()
                        total_penalty[35] += (concrete_log_prob * log_resource_penalty).sum()
                elif self._method == 'reparametrization':
                    if cell.reduction:
                        total_penalty[25] += resource_penalty.sum()
                        total_penalty[38] += log_resource_penalty.sum()
                    else:
                        total_penalty[26] += resource_penalty.sum()
                        total_penalty[37] += log_resource_penalty.sum()
                elif self._method == 'discrete':
                    if cell.reduction:
                        total_penalty[27] += resource_penalty.sum()
                        total_penalty[40] += log_resource_penalty.sum()
                    else:
                        total_penalty[28] += resource_penalty.sum()
                        total_penalty[39] += log_resource_penalty.sum()
                else:
                    total_penalty[-1] += resource_penalty.sum()
                    total_penalty[-2] += resource_penalty.sum()

                concrete_log_prob = self.logp(log_alpha, weights)
                if cell.reduction:
                    # baseline
                    total_penalty[8] += resource_size_baseline.sum()
                    total_penalty[9] += resource_flops_baseline.sum()
                    total_penalty[10] += resource_mac_baseline.sum()
                    total_penalty[24] += resource_penalty.sum()
                    total_penalty[59] += log_resource_penalty.sum()
                    # MC(R)
                    total_penalty[32] += resource_size.sum()
                    total_penalty[33] += resource_flops.sum()
                    total_penalty[34] += resource_mac.sum()
                    # log(|R|)
                    total_penalty[44] += log_resource_size.sum()
                    total_penalty[45] += log_resource_flops.sum()
                    total_penalty[46] += log_resource_mac.sum()
                    # logP * R
                    total_penalty[50] += (concrete_log_prob * resource_size.sum(-1)).sum()
                    total_penalty[51] += (concrete_log_prob * resource_flops.sum(-1)).sum()
                    total_penalty[52] += (concrete_log_prob * resource_mac.sum(-1)).sum()
                    # logP * log(|R|)
                    total_penalty[56] += (concrete_log_prob * log_resource_size).sum()
                    total_penalty[57] += (concrete_log_prob * log_resource_flops).sum()
                    total_penalty[58] += (concrete_log_prob * log_resource_mac).sum()
                    # R - avg
                    total_penalty[63] += resource_size_minus_average.sum()
                    total_penalty[64] += resource_flops_minus_average.sum()
                    total_penalty[65] += resource_mac_minus_average.sum()
                    # logR - log(avg)
                    total_penalty[69] += log_resource_size_minus_average.sum()
                    total_penalty[70] += log_resource_flops_minus_average.sum()
                    total_penalty[71] += log_resource_mac_minus_average.sum()
                    # operation
                    op_reduce[0] += op_size
                    op_reduce[1] += op_flops
                    op_reduce[2] += op_mac

                else:
                    # baseline
                    total_penalty[3] += resource_size_baseline.sum()
                    total_penalty[5] += resource_flops_baseline.sum()
                    total_penalty[6] += resource_mac_baseline.sum()
                    total_penalty[23] += resource_penalty.sum()
                    # MC(R)
                    total_penalty[29] += resource_size.sum()
                    total_penalty[30] += resource_flops.sum()
                    total_penalty[31] += resource_mac.sum()
                    # log(abs(R))
                    total_penalty[41] += log_resource_size.sum()
                    total_penalty[42] += log_resource_flops.sum()
                    total_penalty[43] += log_resource_mac.sum()
                    # logP * R
                    total_penalty[47] += (concrete_log_prob * resource_size.sum(-1)).sum()
                    total_penalty[48] += (concrete_log_prob * resource_flops.sum(-1)).sum()
                    total_penalty[49] += (concrete_log_prob * resource_mac.sum(-1)).sum()
                    # logP * log(|R|)
                    total_penalty[53] += (concrete_log_prob * log_resource_size).sum()
                    total_penalty[54] += (concrete_log_prob * log_resource_flops).sum()
                    total_penalty[55] += (concrete_log_prob * log_resource_mac).sum()
                    # R - avg
                    total_penalty[60] += resource_size_minus_average.sum()
                    total_penalty[61] += resource_flops_minus_average.sum()
                    total_penalty[62] += resource_mac_minus_average.sum()
                    # logR - log(avg)
                    total_penalty[66] += log_resource_size_minus_average.sum()
                    total_penalty[67] += log_resource_flops_minus_average.sum()
                    total_penalty[68] += log_resource_mac_minus_average.sum()
                    # operation
                    op_normal[0] += op_size
                    op_normal[1] += op_flops
                    op_normal[2] += op_mac
        if self.args.snas:
            out = self.global_pooling(s1)
            logits = self.classifier(out.view(out.size(0), -1))
            return logits, logits_aux, total_penalty, op_normal, op_reduce 
        else:
            return logits, error_loss, loss_alpha, total_penalty  

    
    def _discrete_prob(self, log_alpha):
        discrete_prob_1 = F.softmax(log_alpha, dim=-1)
        discrete_prob_0 = 1 - discrete_prob_1
        return discrete_prob_0, discrete_prob_1

    def _arch_entropy(self, log_alpha):
        discrete_prob = F.softmax(log_alpha, dim=-1)
        epsilon = 1e-4
        discrete_prob = torch.max(discrete_prob, torch.ones_like(discrete_prob) * epsilon)
        arch_entropy = -(discrete_prob * torch.log(discrete_prob)).sum()
        return arch_entropy

    def _get_categ_mask(self, log_alpha):
        # log_alpha 2d one_hot 2d
        u = torch.zeros_like(log_alpha).uniform_()
        softmax = torch.nn.Softmax(-1)
        one_hot = softmax((log_alpha + (-((-(u.log())).log()))) / self._temp)
        return one_hot

    def _get_onehot_mask(self, log_alpha):
        if self.args.random_sample:
            uni = torch.ones_like(log_alpha)
            m = torch.distributions.one_hot_categorical.OneHotCategorical(uni)
            one_hot = m.sample()
            return one_hot
        else:
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=F.softmax(log_alpha, dim=-1))
        return m.sample()

    def _get_weights(self, log_alpha):
        if self.args.dsnas or self.args.random_sample and not self.args.random_sample_fix_temp:
            return self._get_onehot_mask(log_alpha)
        else:
            return self._get_categ_mask(log_alpha)

    def arch_parameters(self):
        return self._arch_parameters
    
    def load_child_state_dict(self, model_child):
        model_dict = self.state_dict()
        model_child.load_state_dict(model_dict, strict=False)

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != Self.Primitives.index('none')))[:2]
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((self.Primitives[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.normal_log_alpha, dim=-1).detach().cpu().numpy())
        gene_reduce = _parse(F.softmax(self.reduce_log_alpha, dim=-1).detach().cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def genotype_edge_all(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0

            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != self.Primitives.index('none')))[:2]
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((self.Primitives[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.normal_log_alpha, dim=-1).detach().cpu().numpy())
        gene_reduce = _parse(F.softmax(self.reduce_log_alpha, dim=-1).detach().cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def genotype_child(self, normal_weights, reduce_weights):

        def _parse(weights):
            gene = []
            n = 2
            start = 0

            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != self.Primitives.index('none')))[:2]
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((self.Primitives[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(normal_weights.detach().cpu().numpy())
        gene_reduce = _parse(reduce_weights.detach().cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype



