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
import torch.nn.functional as F
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
parser.add_argument('--gpu_num', type=int, default=0, help='gpu device number')
parser.add_argument('--fix_cudnn', action='store_true', default=False, help='true if fixing cudnn')
parser.add_argument('--resume', action='store_true', default=False, help='reload pretrain model')
parser.add_argument('--resume_path', type=str, default='..', help='the path used to reload model')
parser.add_argument('--resume_epoch', type=int, default=0, help='retrain from num of training epochs')

parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')

parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')

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

parser.add_argument('--bn_track_running_stats', action='store_true', default=True, help='this module tracks the running mean and variance, and when set to False, this module does not track such statistics and always uses batch statistics in both training and eval modes')
parser.add_argument('--iter_reward_norm_num', type=int, default=20, help='iterations to calculate reward') 
parser.add_argument('--share_arch', action='store_true', default=False, help='infer for each train iteration')
parser.add_argument('--child_reward_stat', action='store_true', default=False, help='cal child reward mean and var')
parser.add_argument('--fix_arch', action='store_true', default=False, help='fix child graph architecture')

parser.add_argument('--current_reward', action='store_true', default=False, help='print current reward mean')

parser.add_argument('--order', action='store_true', default=False, help='true if change order')
parser.add_argument('--order2', action='store_true', default=False, help='true if change order')
parser.add_argument('--order3', action='store_true', default=False, help='true if change order')

parser.add_argument('--fix_edge0', action='store_true', default=False, help='fix edge 0')
parser.add_argument('--fix_edge0_op2', action='store_true', default=False, help='fix edge 0')
parser.add_argument('--fix_edge0_op7', action='store_true', default=False, help='fix edge 0')
parser.add_argument('--fix_edge1', action='store_true', default=False, help='fix edge 1')
parser.add_argument('--fix_edge1_op1', action='store_true', default=False, help='fix edge 1')
parser.add_argument('--fix_edge1_op2', action='store_true', default=False, help='fix edge 2')
parser.add_argument('--fix_edge1_op7', action='store_true', default=False, help='fix edge 7')
parser.add_argument('--fix_edge2', action='store_true', default=False, help='fix edge 2')
parser.add_argument('--fix_edge2_op2', action='store_true', default=False, help='fix edge 0')
parser.add_argument('--fix_edge2_op7', action='store_true', default=False, help='fix edge 0')
parser.add_argument('--fix_edge3', action='store_true', default=False, help='fix edge 3')
parser.add_argument('--fix_edge3_op2', action='store_true', default=False, help='fix edge 0')
parser.add_argument('--fix_edge3_op7', action='store_true', default=False, help='fix edge 0')
parser.add_argument('--fix_edge4_op7', action='store_true', default=False, help='fix edge 0')
parser.add_argument('--fix_edge0_conv', action='store_true', default=False, help='fix edge 0 conv')
parser.add_argument('--fix_edge0_noconv', action='store_true', default=False, help='fix edge 0 no conv')
parser.add_argument('--fix_edge0_nopoolskip', action='store_true', default=False, help='fix edge 0 nopoolskip')
parser.add_argument('--fix_edge0_noavgpoolskip', action='store_true', default=False, help='fix edge 0 no avgpool skip')
parser.add_argument('--fix_edge0_nomaxpoolskip', action='store_true', default=False, help='fix edge 0 no avgpool skip')
parser.add_argument('--fix_edge0_noop456', action='store_true', default=False, help='fix edge 0 no op 456')
parser.add_argument('--fix_edge1_conv', action='store_true', default=False, help='fix edge 0 conv')
parser.add_argument('--fix_edge1_noconv', action='store_true', default=False, help='fix edge 0 no conv')
parser.add_argument('--fix_edge1_nonone', action='store_true', default=False, help='fix edge 0 no conv')
parser.add_argument('--fix_edge1_nopoolskip', action='store_true', default=False, help='fix edge 0 nopoolskip')
parser.add_argument('--fix_edge1_noavgpoolskip', action='store_true', default=False, help='fix edge 0 no avgpool skip')
parser.add_argument('--fix_edge1_nomaxpoolskip', action='store_true', default=False, help='fix edge 0 no avgpool skip')
parser.add_argument('--fix_edge1_noop456', action='store_true', default=False, help='fix edge 0 no op 456')
parser.add_argument('--fix_edge3_conv', action='store_true', default=False, help='fix edge 0 conv')
parser.add_argument('--fix_edge3_noconv', action='store_true', default=False, help='fix edge 0 no conv')
parser.add_argument('--fix_edge3_nopoolskip', action='store_true', default=False, help='fix edge 0 nopoolskip')
parser.add_argument('--fix_edge3_noavgpoolskip', action='store_true', default=False, help='fix edge 0 no avgpool skip')
parser.add_argument('--fix_edge3_nomaxpoolskip', action='store_true', default=False, help='fix edge 0 no avgpool skip')
parser.add_argument('--fix_edge3_noop456', action='store_true', default=False, help='fix edge 0 no op 456')
parser.add_argument('--fix_edge4_noconv', action='store_true', default=False, help='fix edge 4 noconv')
parser.add_argument('--fix_edge4_nopoolskip', action='store_true', default=False, help='fix edge 4 noconv')
parser.add_argument('--fix_edge4_noconvskip', action='store_true', default=False, help='fix edge 4 noconv')
parser.add_argument('--fix_edge4_noskip', action='store_true', default=False, help='fix edge 4 noconv')
parser.add_argument('--fix_edge4_noavgpoolskip', action='store_true', default=False, help='fix edge 4 no avgpool skip')
parser.add_argument('--fix_edge4_nomaxpoolskip', action='store_true', default=False, help='fix edge 4 no avgpool skip')
parser.add_argument('--fix_edge4_noop456', action='store_true', default=False, help='fix edge 4 no op 456')

parser.add_argument('--del_Noneinedge0', action='store_true', default=False, help='fix child graph architecture')
parser.add_argument('--del_edge0', action='store_true', default=False, help='del edge 0')
parser.add_argument('--del_edge1', action='store_true', default=False, help='del edge 1')
parser.add_argument('--del_edge2', action='store_true', default=False, help='del edge 2')
parser.add_argument('--del_edge3', action='store_true', default=False, help='del edge 3')
parser.add_argument('--del_edge4', action='store_true', default=False, help='del edge 4')

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
parser.add_argument('--fix_weight', action='store_true', default=False, help='fix the weight parameters')
parser.add_argument('--edge_reward', action='store_true', default=False, help='edge reward normalization')
parser.add_argument('--edge_reward_norm', action='store_true', default=False, help='edge reward normalization')
parser.add_argument('--steps', type=int, default=4, help='steps in each cell')
parser.add_argument('--multiplier', type=int, default=4, help='steps in each cell')

# Sampling settings
parser.add_argument('--temp', type=float, default=1, help='initial temperature(beta)')
parser.add_argument('--temp_min', type=float, default=0.03, help='minimal temperature(beta)')
parser.add_argument('--temp_annealing', action='store_true', default=False, help='true if using temp annealing scheduler')
parser.add_argument('--fix_temp', action='store_true', default=True, help='true if temperature is fixed')
parser.add_argument('--nsample', type=int, default=1, help='child graph sampling times for one batch')
parser.add_argument('--resample_layer', action='store_true', default=False, help='true if resample at each layer')
parser.add_argument('--random_sample', action='store_true', default=False, help='true if sample randomly')
parser.add_argument('--loc_mean', type=float, default=1, help='initial mean value to generate the location')
parser.add_argument('--loc_std', type=float, default=0.01, help='initial std to generate the location')

parser.add_argument('--ckpt_false_list', type=str, default='[]', help='include layers where ckpt is False')
parser.add_argument('--use_ckpt', action='store_true', default=False, help='true if layers out of ckpt_false_list use ckpt')

parser.add_argument('--remark', type=str, default='none', help='further experiment details')
parser.add_argument('--remark_fur', type=str, default='none', help='further experiment details')
args = parser.parse_args()

if args.snas:
    remark = 'snas_'
elif args.dsnas:
    remark = 'dsnas_'

if args.order:
    remark += 'order_'
elif args.order2:
    remark += 'order2_'

# remark += 'epo_' + str(args.epochs) + '_layer_' + str(args.layers) + '_batch_' + str(args.batch_size) + '_drop_prob_' + str(args.drop_path_prob) + '_seed_' + str(args.seed) + '_base_error_gpu_' + str(args.gpu_num)

remark += 'epo_' + str(args.epochs) + '_layer_' + str(args.layers) + '_seed_' + str(args.seed) + '_steps_' + str(args.steps) + '_multi_' + str(args.multiplier)


if args.snas:
    remark += '_temp_' + str(args.temp) + '_temp_min_' + str(args.temp_min)  + '_temp_anneal_' + str(args.temp_annealing)

if args.random_sample:
    remark += '_random_sample'

if args.fix_weight:
    remark += '_fix_weight'

if args.auxiliary:
    remark += '_auxiliary'

if args.remark_fur != 'none':
    remark += '_'+args.remark_fur

args.save = 'search-{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), remark)
args.save_log = 'nas-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), remark)



from scipy.io import loadmat
from torch.utils.data.sampler import SubsetRandomSampler

from torch.autograd import Variable
from model_search import Network

import tensorboardX
import pdb
from datetime import datetime

log_format = '%(asctime)s %(message)s'

CIFAR_CLASSES = 10

generate_date = str(datetime.now().date())


class neural_architecture_search():
    def __init__(self, args):
        self.args = args

        if not torch.cuda.is_available():
            logging.info('no gpu device available')
            sys.exit(1)

        torch.cuda.set_device(self.args.gpu)
        self.device = torch.device("cuda")
        self.rank = 0
        self.seed = self.args.seed
        self.world_size = 1

        if self.args.fix_cudnn:
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
            self.logger = tensorboardX.SummaryWriter('./runs/' + generate_date + '/' + self.args.save_log)
        else:
            self.logger = None

        #initialize loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        #initialize model
        self.init_model()
        if self.args.resume:
            self.reload_model()
        
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
        
        self.model = Network(self.args.init_channels, CIFAR_CLASSES, self.args.layers, self.criterion, self.args, self.rank, self.world_size, self.args.steps, self.args.multiplier)
        self.model.to(self.device)
        for v in self.model.parameters():
            if v.requires_grad:
                if v.grad is None:
                    v.grad = torch.zeros_like(v)
        self.model.normal_log_alpha.grad = torch.zeros_like(self.model.normal_log_alpha)
        self.model.reduce_log_alpha.grad = torch.zeros_like(self.model.reduce_log_alpha)        

    def reload_model(self):
        self.model.load_state_dict(torch.load(self.args.resume_path+'/weights.pt'), strict=True)
 
    def init_optimizer(self):

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=args.weight_decay
        )

        self.arch_optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=self.args.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=self.args.arch_weight_decay
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

        num_train = len(train_data)
        indices = list(range(num_train))

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
            if self.args.child_reward_stat:
                self.update_theta = False
                self.update_alpha = False

            if self.args.current_reward:
                self.model.normal_reward_mean = torch.zeros_like(self.model.normal_reward_mean)
                self.model.reduce_reward_mean = torch.zeros_like(self.model.reduce_reward_mean)
                self.model.count = 0
 
            if epoch < self.args.resume_epoch:
                continue
            self.scheduler.step()
            if self.args.temp_annealing:
                self.model._temp = self.temp_scheduler.step()
            self.lr = self.scheduler.get_lr()[0]
            
            if self.rank == 0:
                logging.info('epoch %d lr %e temp %e', epoch, self.lr, self.model._temp)
                self.logger.add_scalar('epoch_temp', self.model._temp, epoch)
                logging.info(self.model.normal_log_alpha)
                logging.info(self.model.reduce_log_alpha)
                logging.info(F.softmax(self.model.normal_log_alpha, dim=-1))
                logging.info(F.softmax(self.model.reduce_log_alpha, dim=-1))   
      

            genotype_edge_all = self.model.genotype_edge_all()

            if self.rank == 0:
                logging.info('genotype_edge_all = %s', genotype_edge_all)
                # create genotypes.txt file
                txt_name = remark + '_genotype_edge_all_epoch' + str(epoch)
                utils.txt('genotype', self.args.save, txt_name, str(genotype_edge_all), generate_date)

            self.model.train() 
            train_acc, loss, error_loss, loss_alpha = self.train(epoch, logging)
            if self.rank == 0:
                logging.info('train_acc %f', train_acc)
                self.logger.add_scalar("epoch_train_acc", train_acc, epoch)
                self.logger.add_scalar("epoch_train_error_loss", error_loss, epoch)
                if self.args.dsnas:
                    self.logger.add_scalar("epoch_train_alpha_loss", loss_alpha, epoch)

                if self.args.dsnas and not self.args.child_reward_stat:
                    if self.args.current_reward:
                        logging.info('reward mean stat')
                        logging.info(self.model.normal_reward_mean)
                        logging.info(self.model.reduce_reward_mean)
                        logging.info('count')
                        logging.info(self.model.count)
                    else:
                        logging.info('reward mean stat')
                        logging.info(self.model.normal_reward_mean)
                        logging.info(self.model.reduce_reward_mean)
                        if self.model.normal_reward_mean.size(0) > 1:
                            logging.info('reward mean total stat')
                            logging.info(self.model.normal_reward_mean.sum(0))
                            logging.info(self.model.reduce_reward_mean.sum(0))

                if self.args.child_reward_stat:
                    logging.info('reward mean stat')
                    logging.info(self.model.normal_reward_mean.sum(0))
                    logging.info(self.model.reduce_reward_mean.sum(0))
                    logging.info('reward var stat')
                    logging.info(self.model.normal_reward_mean_square.sum(0)-self.model.normal_reward_mean.sum(0)**2)
                    logging.info(self.model.reduce_reward_mean_square.sum(0)-self.model.reduce_reward_mean.sum(0)**2)
               
            
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

        normal_loss_gradient = 0
        reduce_loss_gradient = 0
        normal_total_gradient = 0
        reduce_total_gradient = 0
        
        loss_alpha = None

        train_correct_count = 0
        train_correct_cost = 0
        train_correct_entropy = 0
        train_correct_loss = 0
        train_wrong_count = 0
        train_wrong_cost = 0
        train_wrong_entropy = 0
        train_wrong_loss = 0

        count = 0
        for step, (input, target) in enumerate(self.train_queue):
 
            n = input.size(0)
            input = input.to(self.device)
            target = target.to(self.device, non_blocking=True)
            if self.args.snas:
                logits, logits_aux = self.model(input)
                error_loss = self.criterion(logits, target)
                if self.args.auxiliary:
                    loss_aux = self.criterion(logits_aux, target)
                    error_loss += self.args.auxiliary_weight*loss_aux

            if self.args.dsnas:
                logits, error_loss, loss_alpha = self.model(input, target, self.criterion, update_theta=self.update_theta, update_alpha=self.update_alpha)

            for i in range(logits.size(0)):
                index = logits[i].topk(5,0,True, True)[1]
                if  index[0].item() == target[i].item():
                    train_correct_cost += (-logits[i, target[i].item()] + (F.softmax(logits[i])*logits[i]).sum())
                    train_correct_count += 1
                    discrete_prob = F.softmax(logits[i], dim=-1)
                    train_correct_entropy += -(discrete_prob * torch.log(discrete_prob)).sum(-1)
                    train_correct_loss += -torch.log(discrete_prob)[target[i].item()]
                else:
                    train_wrong_cost += (-logits[i, target[i].item()] + (F.softmax(logits[i])*logits[i]).sum())
                    train_wrong_count += 1
                    discrete_prob = F.softmax(logits[i], dim=-1)
                    train_wrong_entropy += -(discrete_prob * torch.log(discrete_prob)).sum(-1)
                    train_wrong_loss += -torch.log(discrete_prob)[target[i].item()]
            
            num_normal = self.model.num_normal
            num_reduce = self.model.num_reduce

            if self.args.snas or self.args.dsnas:
                loss = error_loss.clone()          
 
            #self.update_lr()

            # logging gradient
            count += 1
            if self.args.snas:
                self.optimizer.zero_grad()
                self.arch_optimizer.zero_grad()
                error_loss.backward(retain_graph=True)
                if not self.args.random_sample:
                    normal_loss_gradient += self.model.normal_log_alpha.grad
                    reduce_loss_gradient += self.model.reduce_log_alpha.grad
                self.optimizer.zero_grad()
                self.arch_optimizer.zero_grad()

            if self.args.snas and (not self.args.random_sample and not self.args.dsnas):
                loss.backward()

            if not self.args.random_sample:
                normal_total_gradient += self.model.normal_log_alpha.grad
                reduce_total_gradient += self.model.reduce_log_alpha.grad

            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            arch_grad_norm = nn.utils.clip_grad_norm_(self.model.arch_parameters(), 10.)
            
            grad.update(arch_grad_norm)
            if not self.args.fix_weight and self.update_theta:
                self.optimizer.step()
            self.optimizer.zero_grad() 
           
            if not self.args.random_sample and self.update_alpha:
                self.arch_optimizer.step()
            self.arch_optimizer.zero_grad()
            
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

            objs.update(error_loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.args.report_freq == 0 and self.rank == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                self.logger.add_scalar("iter_train_top1_acc", top1.avg, step + len(self.train_queue.dataset) * epoch)

        if self.rank == 0:
            logging.info('-------loss gradient--------')
            logging.info(normal_loss_gradient / count)
            logging.info(reduce_loss_gradient / count)
            logging.info('-------total gradient--------')
            logging.info(normal_total_gradient / count)
            logging.info(reduce_total_gradient / count)

        logging.info('correct loss ')
        logging.info((train_correct_loss/train_correct_count).item())
        logging.info('correct entropy ')
        logging.info((train_correct_entropy/train_correct_count).item())
        logging.info('correct cost ')
        logging.info((train_correct_cost/train_correct_count).item())
        logging.info('correct count ')
        logging.info(train_correct_count)

        logging.info('wrong loss ')
        logging.info((train_wrong_loss/train_wrong_count).item())
        logging.info('wrong entropy ')
        logging.info((train_wrong_entropy/train_wrong_count).item())
        logging.info('wrong cost ')
        logging.info((train_wrong_cost/train_wrong_count).item())
        logging.info('wrong count ')
        logging.info(train_wrong_count)

        logging.info('total loss ')
        logging.info(((train_correct_loss+train_wrong_loss)/(train_correct_count+train_wrong_count)).item())
        logging.info('total entropy ')
        logging.info(((train_correct_entropy+train_wrong_entropy)/(train_correct_count+train_wrong_count)).item())
        logging.info('total cost ')
        logging.info(((train_correct_cost+train_wrong_cost)/(train_correct_count+train_wrong_count)).item())
        logging.info('total count ')
        logging.info(train_correct_count+train_wrong_count)

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
                    logits, logits_aux = self.model(input)
                    loss = self.criterion(logits, target)
                elif self.args.dsnas:
                    logits, error_loss, loss_alpha = self.model(input, target, self.criterion)
                    loss = error_loss

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

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
