import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
    
parser = argparse.ArgumentParser("cifar")

# General settings
parser.add_argument('--snas', action='store_true', default=False, help='true if using snas model')
parser.add_argument('--dsnas', action='store_true', default=False, help='true if using dsnas')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument("--local_rank", type=int)
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')

# Training settings
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--distributed', action='store_true', default=False, help='true if using multi-GPU training')
parser.add_argument('--fix_seedcudnn', action='store_true', default=False, help='true if fixing cudnn')
parser.add_argument('--port', type=int, default=23333, help='distributed port')

parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')

parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')

parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

parser.add_argument('--ema_eta', type=float, default=1e-3, help='eta for exponential moving average')

parser.add_argument('--bn_affine', action='store_true', default=False, help='update para in BatchNorm or not')

# Network settings
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary loss')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--gen_max_child', action='store_true', default=False, help='generate child network by argmax(alpha)')
parser.add_argument('--gen_max_child_flag', action='store_true', default=False, help='flag of generating child network by argmax(alpha)')
parser.add_argument('--prox_policy_opt', action='store_true', default=False, help='use proximal policy optimization')
parser.add_argument('--prox_policy_epi', type=float, default=2e-1, help='use proximal policy epison')
parser.add_argument('--fix_weight', action='store_true', default=False, help='fix the weight parameters')
parser.add_argument('--edge_reward', action='store_true', default=False, help='edge reward normalization')
parser.add_argument('--edge_reward_norm', action='store_true', default=False, help='edge reward normalization')
parser.add_argument('--edge_reward_norm_mean_0', action='store_true', default=False, help='edge reward normalization')
parser.add_argument('--add_entropy_loss', action='store_true', default=False, help='add entropy loss')
parser.add_argument('--alternate_update', action='store_true', default=False, help='add entropy loss')
parser.add_argument('--enas_reward', action='store_true', default=False, help='enas reward')
parser.add_argument('--enas_reward_norm', action='store_true', default=False, help='enas reward norm')

# Sampling settings
parser.add_argument('--temp', type=float, default=1, help='initial temperature(beta)')
parser.add_argument('--temp_min', type=float, default=0.03, help='minimal temperature(beta)')
parser.add_argument('--temp_annealing', action='store_true', default=False, help='true if using temp annealing scheduler')
parser.add_argument('--fix_temp', action='store_true', default=True, help='true if temperature is fixed')
parser.add_argument('--nsample', type=int, default=1, help='child graph sampling times for one batch')
parser.add_argument('--resample_layer', action='store_true', default=False, help='true if resample at each layer')
parser.add_argument('--random_sample', action='store_true', default=False, help='true if sample randomly')
parser.add_argument('--random_sample_fix_temp', action='store_true', default=False, help='true if sample randomly with fixed temp')
parser.add_argument('--random_sample_pretrain', action='store_true', default=False, help='true if using random sample pretrain')
parser.add_argument('--random_sample_pretrain_epoch', type=int, default=50, help='child graph sampling pretrain epochs')
parser.add_argument('--loc_mean', type=float, default=1, help='initial mean value to generate the location')
parser.add_argument('--loc_std', type=float, default=0.01, help='initial std to generate the location')

# Resource constraint settings
parser.add_argument('--resource_efficient', action='store_true', default=False,
                    help='true if adding resource constraint')
parser.add_argument('--method', type=str, default='policy_gradient',
                    help='three methods to calculate expectation of resources: policy_gradient reparametrization discrete')
parser.add_argument('--normalization', action='store_true', default=False, help='true if using normalized resource')
parser.add_argument('--running_mean_var', action='store_true', default=False,
                    help='true if using Running Mean Variance in the normalization of resource')
default_lambda = 1e10
parser.add_argument('--resource_lambda', type=float, default=default_lambda, help='learning rate for resource-efficient arch encoding')
parser.add_argument('--separation', type=str, default='all', help='calculate three resources separately: flops, mac, and size')
parser.add_argument('--log_penalty', action='store_true', default=False, help='true if take log on penalty')
parser.add_argument('--loss', action='store_true', default=False, help='true if add loss')
parser.add_argument('--minus_baseline', action='store_true', default=False, help='true if resource minus baseline')
parser.add_argument('--ratio', type=float, default=1, help='resource lambda reduction/normal')
# resource scheduler
parser.add_argument('--resource_sche', action='store_true', default=False, help='true if add resource_scheduler')
parser.add_argument('--lambda_constant', type=float, default=1e-4, help='constant resource lambda')
parser.add_argument('--slope_flag', type=float, default=0.05, help='flag of valis_acc-epoch')
parser.add_argument('--mavg_alpha', type=float, default=0.5, help='alpha of moving avg to smooth valid_acc-epoch')
parser.add_argument('--epoch_flag_add', type=float, default=45, help='specified epoch to add lambda_constant')
parser.add_argument('--epoch_flag_rm', type=float, default=90, help='specified epoch to remove added lambda_constant')
# distributed settings
parser.add_argument('--ckpt_false_list', type=str, default='[]', help='include layers where ckpt is False')
parser.add_argument('--use_ckpt', action='store_true', default=False, help='true if layers out of ckpt_false_list use ckpt')

parser.add_argument('--remark', type=str, default='none', help='further experiment details')
args = parser.parse_args()

args.save = 'search-{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), args.remark)


if args.distributed:
    import torch.distributed as dist
    from dist_util_torch import init_dist, broadcast_params, reduce_gradients, reduce_tensorgradients, part_reduce_gradients

from scipy.io import loadmat
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from torch.autograd import Variable
from model_search import Network

import tensorboardX
import pdb
from datetime import datetime
#from scipy.linalg import null_space

log_format = '%(asctime)s %(message)s'

CIFAR_CLASSES = 10

generate_date = str(datetime.now().date())


class neural_architecture_search():
    def __init__(self, args):
        self.args = args

        if not torch.cuda.is_available():
            logging.info('no gpu device available')
            sys.exit(1)

        if self.args.distributed:
            # Init distributed environment
            self.rank, self.world_size, self.device = init_dist(port=self.args.port)
            self.seed = self.rank * self.args.seed
        else: 
            torch.cuda.set_device(self.args.gpu)
            self.device = torch.device("cuda")
            self.rank = 0
            self.seed = self.args.seed
            self.world_size = 1

        if self.args.fix_seedcudnn:
            random.seed(self.seed)
            torch.backends.cudnn.deterministic = True
            np.random.seed(self.seed)
            cudnn.benchmark = False
            torch.manual_seed(self.seed)
            cudnn.enabled = True
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        else:
            np.random.seed(self.seed)
            cudnn.benchmark =True
            torch.manual_seed(self.seed)
            cudnn.enabled = True
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        self.path = os.path.join(generate_date, self.args.save)
        if self.rank == 0:
            utils.create_exp_dir(generate_date, self.path, scripts_to_save=glob.glob('*.py'))
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(self.path, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            logging.info("self.args = %s", self.args)
            self.logger = tensorboardX.SummaryWriter('./runs/' + generate_date + '/nas_{}'.format(self.args.remark))
        else:
            self.logger = None

        # set default resource_lambda for different methods
        if self.args.resource_efficient:
            if self.args.method == 'policy_gradient':
                if self.args.log_penalty:
                    default_resource_lambda = 1e-4
                else:
                    default_resource_lambda = 1e-5
            if self.args.method == 'reparametrization':
                if self.args.log_penalty:
                    default_resource_lambda = 1e-2
                else:
                    default_resource_lambda = 1e-5
            if self.args.method == 'discrete':
                if self.args.log_penalty:
                    default_resource_lambda = 1e-2
                else:
                    default_resource_lambda = 1e-4
            if self.args.resource_lambda == default_lambda:
                self.args.resource_lambda = default_resource_lambda

        #initialize loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        #initialize model
        self.init_model()
        
        #calculate model param size
        if self.rank == 0:
            logging.info("param size = %fMB", utils.count_parameters_in_MB(self.model))
            self.model._logger = self.logger
            self.model._logging = logging    
        
        #initialize optimizer
        self.init_optimizer()

        #iniatilize dataset loader
        self.init_loaddata()
        
        self.update_theta = True
        self.update_alpha = True

    def init_model(self):
        
        self.model = Network(self.args.init_channels, CIFAR_CLASSES, self.args.layers, self.criterion, self.args, self.rank, self.world_size)
        self.model.to(self.device)
        if self.args.distributed:
            broadcast_params(self.model)
        for v in self.model.parameters():
            if v.requires_grad:
                if v.grad is None:
                    v.grad = torch.zeros_like(v)
        self.model.normal_log_alpha.grad = torch.zeros_like(self.model.normal_log_alpha)
        self.model.reduce_log_alpha.grad = torch.zeros_like(self.model.reduce_log_alpha)        

    def init_optimizer(self):

        if args.distributed:
            self.optimizer = torch.optim.SGD(
                [param for name, param in self.model.named_parameters() if name != 'normal_log_alpha' and name != 'reduce_log_alpha'],
                self.args.learning_rate,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
            self.arch_optimizer = torch.optim.Adam(
                [param for name, param in self.model.named_parameters() if name == 'normal_log_alpha' or name == 'reduce_log_alpha'],
                lr=self.args.arch_learning_rate,
                betas=(0.5, 0.999),
                weight_decay=self.args.arch_weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                self.args.learning_rate,
                momentum=self.args.momentum,
                weight_decay=args.weight_decay
            )

            self.arch_optimizer = torch.optim.SGD(
                self.model.arch_parameters(),
                lr=self.args.arch_learning_rate
            )
            

    def init_loaddata(self):

        train_transform, valid_transform = utils._data_transforms_cifar10(self.args)
        train_data = dset.CIFAR10(root=self.args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=self.args.data, train=False, download=True, transform=valid_transform)


        if self.args.seed:
            def worker_init_fn():
                seed = self.seed
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                return
        else: 
            worker_init_fn = None

        if self.args.distributed:
            train_sampler = DistributedSampler(train_data)
            valid_sampler = DistributedSampler(valid_data) 
                
            self.train_queue = torch.utils.data.DataLoader(train_data, batch_size=self.args.batch_size//self.world_size, shuffle=False, num_workers=0, pin_memory=False, sampler=train_sampler)
            self.valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=self.args.batch_size//self.world_size, shuffle=False, num_workers=0, pin_memory=False, sampler=valid_sampler)

        else:
            self.train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size,
                shuffle=True,
                pin_memory=False, num_workers=2)

            self.valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size, shuffle=False, pin_memory=False, num_workers=2)
        

    def main(self):
        # lr scheduler: cosine annealing
        # temp scheduler: linear annealing (self-defined in utils)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(self.args.epochs), eta_min=self.args.learning_rate_min)

        self.temp_scheduler = utils.Temp_Scheduler(self.args.epochs, self.model._temp, self.args.temp, temp_min=self.args.temp_min)

        for epoch in range(self.args.epochs):
            if self.args.random_sample_pretrain:
                if epoch < self.args.random_sample_pretrain_epoch:
                    self.args.random_sample = True
                else:
                    self.args.random_sample = False
                    
            self.scheduler.step()
            if self.args.temp_annealing:
                self.model._temp = self.temp_scheduler.step()
            self.lr = self.scheduler.get_lr()[0]
            
            if self.rank == 0:
                logging.info('epoch %d lr %e temp %e', epoch, self.lr, self.model._temp)
                self.logger.add_scalar('epoch_temp', self.model._temp, epoch)
                logging.info(self.model.normal_log_alpha)
                logging.info(self.model.reduce_log_alpha)
                logging.info(self.model._get_weights(self.model.normal_log_alpha[0]))
                logging.info(self.model._get_weights(self.model.reduce_log_alpha[0]))

            genotype_edge_all = self.model.genotype_edge_all()

            if self.rank == 0:
                logging.info('genotype_edge_all = %s', genotype_edge_all)
                # create genotypes.txt file
                txt_name = self.args.remark + '_genotype_edge_all_epoch' + str(epoch)
                utils.txt('genotype', self.args.save, txt_name, str(genotype_edge_all), generate_date)

            self.model.train() 
            train_acc, loss, error_loss, loss_alpha = self.train(epoch, logging)
            if self.rank == 0:
                logging.info('train_acc %f', train_acc)
                self.logger.add_scalar("epoch_train_acc", train_acc, epoch)
                self.logger.add_scalar("epoch_train_error_loss", error_loss, epoch)
                if self.args.dsnas:
                    self.logger.add_scalar("epoch_train_alpha_loss", loss_alpha, epoch)

            # validation
            self.model.eval()
            valid_acc, valid_obj = self.infer(epoch)
            if self.args.gen_max_child:
                self.args.gen_max_child_flag = True
                valid_acc_max_child, valid_obj_max_child = self.infer(epoch)                
                self.args.gen_max_child_flag = False

            if self.rank == 0:
                logging.info('valid_acc %f', valid_acc)
                self.logger.add_scalar("epoch_valid_acc", valid_acc, epoch)
                if self.args.gen_max_child:
                    logging.info('valid_acc_argmax_alpha %f', valid_acc_max_child)
                    self.logger.add_scalar("epoch_valid_acc_argmax_alpha", valid_acc_max_child, epoch)

                utils.save(self.model, os.path.join(self.path, 'weights.pt'))

        if self.rank == 0:
            logging.info(self.model.normal_log_alpha)
            logging.info(self.model.reduce_log_alpha)
            genotype_edge_all = self.model.genotype_edge_all()
            logging.info('genotype_edge_all = %s', genotype_edge_all)



    def train(self, epoch, logging):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        grad = utils.AvgrageMeter()

        normal_resource_gradient = 0
        reduce_resource_gradient = 0
        normal_loss_gradient = 0
        reduce_loss_gradient = 0
        normal_total_gradient = 0
        reduce_total_gradient = 0
        
        loss_alpha = None

        count = 0
        for step, (input, target) in enumerate(self.train_queue):
            if self.args.alternate_update:
                if step % 2 == 0:
                    self.update_theta = True
                    self.update_alpha = False
                else:
                    self.update_theta = False
                    self.update_alpha = True
                    
            n = input.size(0)
            input = input.to(self.device)
            target = target.to(self.device, non_blocking=True)
            if self.args.snas:
                logits, logits_aux, penalty, op_normal, op_reduce = self.model(input)
                error_loss = self.criterion(logits, target)
                if self.args.auxiliary:
                    loss_aux = self.criterion(logits_aux, target)
                    error_loss += self.args.auxiliary_weight*loss_aux

            if self.args.dsnas:
                logits, error_loss, loss_alpha, penalty = self.model(input, target, self.criterion)
            
            num_normal = self.model.num_normal
            num_reduce = self.model.num_reduce
            normal_arch_entropy = self.model._arch_entropy(self.model.normal_log_alpha)
            reduce_arch_entropy = self.model._arch_entropy(self.model.reduce_log_alpha)
            
            if self.args.resource_efficient:
                if self.args.method == 'policy_gradient':
                    resource_penalty = (penalty[2]) / 6 + self.args.ratio * (penalty[7]) / 2
                    log_resource_penalty = (penalty[35]) / 6 + self.args.ratio * (penalty[36]) / 2
                elif self.args.method == 'reparametrization':
                    resource_penalty = (penalty[26]) / 6 + self.args.ratio * (penalty[25]) / 2
                    log_resource_penalty = (penalty[37]) / 6 + self.args.ratio * (penalty[38]) / 2
                elif self.args.method == 'discrete':
                    resource_penalty = (penalty[28]) / 6 + self.args.ratio * (penalty[27]) / 2
                    log_resource_penalty = (penalty[39]) / 6 + self.args.ratio * (penalty[40]) / 2
                elif self.args.method == 'none':
                    # TODo
                    resource_penalty = torch.zeros(1).cuda()
                    log_resource_penalty = torch.zeros(1).cuda()
                else:
                    logging.info("wrongly input of method, please re-enter --method from 'policy_gradient', 'discrete', "
                                 "'reparametrization', 'none'")
                    sys.exit(1)
            else:
                resource_penalty = torch.zeros(1).cuda()
                log_resource_penalty = torch.zeros(1).cuda()


            if self.args.log_penalty:
                resource_loss = self.model._resource_lambda * log_resource_penalty
            else:
                resource_loss = self.model._resource_lambda * resource_penalty
                

            if self.args.loss:
                if self.args.snas:
                    loss = resource_loss.clone() + error_loss.clone()
                elif self.args.dsnas:
                    loss = resource_loss.clone()
                else:
                    loss = resource_loss.clone() + -child_coef * (torch.log(normal_one_hot_prob) + torch.log(reduce_one_hot_prob)).sum()
            else:
                if self.args.snas or self.args.dsnas:
                    loss = error_loss.clone()
          
 
            if self.args.distributed:
                loss.div_(self.world_size)
                error_loss.div_(self.world_size)
                resource_loss.div_(self.world_size)
                if self.args.dsnas:
                    loss_alpha.div_(self.world_size)
            
            # logging gradient
            count += 1
            if self.args.resource_efficient:
                self.optimizer.zero_grad()
                self.arch_optimizer.zero_grad()
                resource_loss.backward(retain_graph=True)
                if not self.args.random_sample:
                    normal_resource_gradient += self.model.normal_log_alpha.grad
                    reduce_resource_gradient += self.model.reduce_log_alpha.grad
            if self.args.snas:
                self.optimizer.zero_grad()
                self.arch_optimizer.zero_grad()
                error_loss.backward(retain_graph=True)
                if not self.args.random_sample:
                    normal_loss_gradient += self.model.normal_log_alpha.grad
                    reduce_loss_gradient += self.model.reduce_log_alpha.grad
                self.optimizer.zero_grad()
                self.arch_optimizer.zero_grad()

            if self.args.snas or not self.args.random_sample and not self.args.dsnas:
                loss.backward()
            if not self.args.random_sample:
                normal_total_gradient += self.model.normal_log_alpha.grad
                reduce_total_gradient += self.model.reduce_log_alpha.grad

            if self.args.distributed:
                reduce_tensorgradients(self.model.parameters(), sync=True)
                nn.utils.clip_grad_norm_(
                    [param for name, param in self.model.named_parameters() if name != 'normal_log_alpha' and name != 'reduce_log_alpha'],
                    self.args.grad_clip
                )
                arch_grad_norm = nn.utils.clip_grad_norm_(
                    [param for name, param in self.model.named_parameters() if name == 'normal_log_alpha' or name == 'reduce_log_alpha'], 10.
                )
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                arch_grad_norm = nn.utils.clip_grad_norm_(self.model.arch_parameters(), 10.)

            grad.update(arch_grad_norm)
            if not self.args.fix_weight and self.update_theta:
                self.optimizer.step()
            self.optimizer.zero_grad() 
            if not self.args.random_sample and self.update_alpha:
                self.arch_optimizer.step()
            self.arch_optimizer.zero_grad()

            if self.rank == 0:
                self.logger.add_scalar("iter_train_loss", error_loss, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("normal_arch_entropy", normal_arch_entropy, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("reduce_arch_entropy", reduce_arch_entropy, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("total_arch_entropy", normal_arch_entropy+reduce_arch_entropy, step + len(self.train_queue.dataset) * epoch)
                if self.args.dsnas:
                    #reward_normal_edge
                    self.logger.add_scalar("reward_normal_edge_0", self.model.normal_edge_reward[0], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_normal_edge_1", self.model.normal_edge_reward[1], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_normal_edge_2", self.model.normal_edge_reward[2], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_normal_edge_3", self.model.normal_edge_reward[3], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_normal_edge_4", self.model.normal_edge_reward[4], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_normal_edge_5", self.model.normal_edge_reward[5], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_normal_edge_6", self.model.normal_edge_reward[6], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_normal_edge_7", self.model.normal_edge_reward[7], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_normal_edge_8", self.model.normal_edge_reward[8], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_normal_edge_9", self.model.normal_edge_reward[9], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_normal_edge_10", self.model.normal_edge_reward[10], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_normal_edge_11", self.model.normal_edge_reward[11], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_normal_edge_12", self.model.normal_edge_reward[12], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_normal_edge_13", self.model.normal_edge_reward[13], step + len(self.train_queue.dataset) * epoch)
                    #reward_reduce_edge
                    self.logger.add_scalar("reward_reduce_edge_0", self.model.reduce_edge_reward[0], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_reduce_edge_1", self.model.reduce_edge_reward[1], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_reduce_edge_2", self.model.reduce_edge_reward[2], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_reduce_edge_3", self.model.reduce_edge_reward[3], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_reduce_edge_4", self.model.reduce_edge_reward[4], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_reduce_edge_5", self.model.reduce_edge_reward[5], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_reduce_edge_6", self.model.reduce_edge_reward[6], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_reduce_edge_7", self.model.reduce_edge_reward[7], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_reduce_edge_8", self.model.reduce_edge_reward[8], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_reduce_edge_9", self.model.reduce_edge_reward[9], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_reduce_edge_10", self.model.reduce_edge_reward[10], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_reduce_edge_11", self.model.reduce_edge_reward[11], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_reduce_edge_12", self.model.reduce_edge_reward[12], step + len(self.train_queue.dataset) * epoch)
                    self.logger.add_scalar("reward_reduce_edge_13", self.model.reduce_edge_reward[13], step + len(self.train_queue.dataset) * epoch)
                #policy size
                self.logger.add_scalar("iter_normal_size_policy", penalty[2] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_size_policy", penalty[7] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                # baseline: discrete_probability
                self.logger.add_scalar("iter_normal_size_baseline", penalty[3] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_flops_baseline", penalty[5] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_mac_baseline", penalty[6] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_size_baseline", penalty[8] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_flops_baseline", penalty[9] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_mac_baseline", penalty[10] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                # R - median(R)
                self.logger.add_scalar("iter_normal_size-avg", penalty[60] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_flops-avg", penalty[61] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_mac-avg", penalty[62] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_size-avg", penalty[63] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_flops-avg", penalty[64] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_mac-avg", penalty[65] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                # lnR - ln(median)
                self.logger.add_scalar("iter_normal_ln_size-ln_avg", penalty[66] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_ln_flops-ln_avg", penalty[67] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_ln_mac-ln_avg", penalty[68] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_ln_size-ln_avg", penalty[69] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_ln_flops-ln_avg", penalty[70] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_ln_mac-ln_avg", penalty[71] / num_reduce, step + len(self.train_queue.dataset) * epoch)

                '''
                self.logger.add_scalar("iter_normal_size_normalized", penalty[17] / 6, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_flops_normalized", penalty[18] / 6, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_mac_normalized", penalty[19] / 6, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_size_normalized", penalty[20] / 2, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_flops_normalized", penalty[21] / 2, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_mac_normalized", penalty[22] / 2, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_penalty_normalized", penalty[23] / 6,
                                  step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_penalty_normalized", penalty[24] / 2,
                                  step + len(self.train_queue.dataset) * epoch)
                '''
                # Monte_Carlo(R_i)
                self.logger.add_scalar("iter_normal_size_mc", penalty[29] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_flops_mc", penalty[30] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_mac_mc", penalty[31] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_size_mc", penalty[32] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_flops_mc", penalty[33] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_mac_mc", penalty[34] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                # log(|R_i|)
                self.logger.add_scalar("iter_normal_log_size", penalty[41] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_log_flops", penalty[42] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_log_mac", penalty[43] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_log_size", penalty[44] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_log_flops", penalty[45] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_log_mac", penalty[46] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                # log(P)R_i
                self.logger.add_scalar("iter_normal_logP_size", penalty[47] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_logP_flops", penalty[48] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_logP_mac", penalty[49] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_logP_size", penalty[50] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_logP_flops", penalty[51] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_logP_mac", penalty[52] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                # log(P)log(R_i)
                self.logger.add_scalar("iter_normal_logP_log_size", penalty[53] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_logP_log_flops", penalty[54] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_normal_logP_log_mac", penalty[55] / num_normal, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_logP_log_size", penalty[56] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_logP_log_flops", penalty[57] / num_reduce, step + len(self.train_queue.dataset) * epoch)
                self.logger.add_scalar("iter_reduce_logP_log_mac", penalty[58] / num_reduce, step + len(self.train_queue.dataset) * epoch)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

            if self.args.distributed:
                loss = loss.detach()
                dist.all_reduce(error_loss)
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1.div_(self.world_size)
                prec5.div_(self.world_size)
                #dist_util.all_reduce([loss, prec1, prec5], 'mean')
            objs.update(error_loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.args.report_freq == 0 and self.rank == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                self.logger.add_scalar("iter_train_top1_acc", top1.avg, step + len(self.train_queue.dataset) * epoch)


        if self.rank == 0:
            logging.info('-------resource gradient--------')
            logging.info(normal_resource_gradient / count)
            logging.info(reduce_resource_gradient / count)
            logging.info('-------loss gradient--------')
            logging.info(normal_loss_gradient / count)
            logging.info(reduce_loss_gradient / count)
            logging.info('-------total gradient--------')
            logging.info(normal_total_gradient / count)
            logging.info(reduce_total_gradient / count)

        return top1.avg, loss, error_loss, loss_alpha

    def infer(self, epoch):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
       
        self.model.eval()
        with torch.no_grad():
            for step, (input, target) in enumerate(self.valid_queue):
                input = input.to(self.device)
                target = target.to(self.device)
                if self.args.snas:
                    logits, logits_aux, resource_loss, op_normal, op_reduce = self.model(input)
                    loss = self.criterion(logits, target)
                elif self.args.dsnas:
                    logits, error_loss, loss_alpha, resource_loss = self.model(input, target, self.criterion)
                    loss = error_loss

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

                if self.args.distributed:
                    loss.div_(self.world_size)
                    loss = loss.detach()
                    dist.all_reduce(loss)
                    dist.all_reduce(prec1)
                    dist.all_reduce(prec5)
                    prec1.div_(self.world_size)
                    prec5.div_(self.world_size)
                objs.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                if step % self.args.report_freq == 0 and self.rank == 0:
                    logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                    self.logger.add_scalar("iter_valid_loss", loss, step + len(self.valid_queue.dataset) * epoch)
                    self.logger.add_scalar("iter_valid_top1_acc", top1.avg, step + len(self.valid_queue.dataset) * epoch)

        return top1.avg, objs.avg


if __name__ == '__main__':
    architecture = neural_architecture_search(args)
    architecture.main()
