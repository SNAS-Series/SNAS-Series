from __future__ import division
import argparse
import os
import glob
import time
from datetime import datetime
import torch.distributed as dist
import numpy as np
import torch
import utils
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml
import sys
from tensorboardX import SummaryWriter
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))

from devkit.core import (init_dist, broadcast_params, average_gradients, load_state_ckpt, load_state, save_checkpoint, LRScheduler, CrossEntropyLoss)
from devkit.dataset.imagenet_dataset import ImagenetDataset
from network import ShuffleNetV2_OneShot

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--SinglePath', action='store_true', default=False, help='true if using SinglePath')
parser.add_argument('--config', default='configs/shufflenet_v2_bn.yaml')
parser.add_argument("--local_rank", type=int)
parser.add_argument(
    '--port', default=29500, type=int, help='port of server')
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--model_dir', type=str)
parser.add_argument('--resume_from', default='', help='resume_from')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--off-ms', action='store_true')

parser.add_argument('--epochs', type=int, default=240, help='num of training epochs')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=24, help='random seed')
parser.add_argument('--loc_mean', type=float, default=1, help='initial mean value to generate the location')
parser.add_argument('--loc_std', type=float, default=0.01, help='initial std to generate the location')
parser.add_argument('--bn_affine', action='store_true', default=False, help='bn affine flag')
parser.add_argument('--bn_eps', type=float, default=1e-2, help='batch normalization epison')
parser.add_argument('--early_fix_arch', action='store_true', default=False, help='bn affine flag')
parser.add_argument('--flops_loss', action='store_true', default=False, help='flops loss flag')
parser.add_argument('--flops_loss_coef', type=float, default=1e-5, help='flops loss coefficient')
parser.add_argument('--pretrain_epoch', type=int, default=0, help='pretrain epochs')
parser.add_argument('--use_dropout', action='store_true', default=False, help='dropout flag')

parser.add_argument('--random_sample', action='store_true', default=False, help='true if sample randomly')
parser.add_argument('--gen_max_child', action='store_true', default=False, help='generate child network by argmax(alpha)')
parser.add_argument('--gen_max_child_flag', action='store_true', default=False, help='flag of generating child network by argmax(alpha)')
parser.add_argument('--remark', type=str, default='none', help='experiment details')

args = parser.parse_args()

def main():
    global args, best_prec1
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('Enabled distributed training.')

    rank, world_size = init_dist(
        backend='nccl', port=args.port)
    args.rank = rank
    args.world_size = world_size

    np.random.seed(args.seed*args.rank)
    torch.manual_seed(args.seed*args.rank)
    torch.cuda.manual_seed(args.seed*args.rank)
    torch.cuda.manual_seed_all(args.seed*args.rank)
    print('random seed: ', args.seed*args.rank)

    # create model
    print("=> creating model '{}'".format(args.model))
    if args.SinglePath:
        architecture = 20*[0]
        channels_scales = 20*[1.0]
        model = ShuffleNetV2_OneShot(args=args, architecture=architecture, channels_scales=channels_scales)
        model.cuda()
        broadcast_params(model)
        for v in model.parameters():
            if v.requires_grad:
                if v.grad is None:
                    v.grad = torch.zeros_like(v)
        model.log_alpha.grad = torch.zeros_like(model.log_alpha)   
    
    criterion = CrossEntropyLoss(smooth_eps=0.1, smooth_dist=(torch.ones(1000)*0.001).cuda()).cuda()


    wo_wd_params = []
    wo_wd_param_names = []
    network_params = []
    network_param_names = []

    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            for key, value in mod.named_parameters():
                wo_wd_param_names.append(name+'.'+key)
        
    for key, value in model.named_parameters():
        if key != 'log_alpha':
            if value.requires_grad:
                if key in wo_wd_param_names:
                    wo_wd_params.append(value)
                else:
                    network_params.append(value)
                    network_param_names.append(key)

    params = [
        {'params': network_params,
         'lr': args.base_lr,
         'weight_decay': args.weight_decay },
        {'params': wo_wd_params,
         'lr': args.base_lr,
         'weight_decay': 0.},
    ]
    param_names = [network_param_names, wo_wd_param_names]
    if args.rank == 0:
        print('>>> params w/o weight decay: ', wo_wd_param_names)

    optimizer = torch.optim.SGD(params, momentum=args.momentum)
    if args.SinglePath:
        arch_optimizer = torch.optim.Adam(
            [param for name, param in model.named_parameters() if name == 'log_alpha'],
            lr=args.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay
        )

    # auto resume from a checkpoint
    remark = 'imagenet_'
    remark += 'epo_' + str(args.epochs) + '_layer_' + str(args.layers) + '_batch_' + str(args.batch_size) + '_lr_' + str(args.base_lr)  + '_seed_' + str(args.seed) + '_pretrain_' + str(args.pretrain_epoch)

    if args.early_fix_arch:
        remark += '_early_fix_arch'  

    if args.flops_loss:
        remark += '_flops_loss_' + str(args.flops_loss_coef)

    if args.remark != 'none':
        remark += '_'+args.remark

    args.save = 'search-{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), remark)
    args.save_log = 'nas-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), remark)
    generate_date = str(datetime.now().date())

    path = os.path.join(generate_date, args.save)
    if args.rank == 0:
        log_format = '%(asctime)s %(message)s'
        utils.create_exp_dir(generate_date, path, scripts_to_save=glob.glob('*.py'))
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(path, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        logging.info("args = %s", args)
        writer = SummaryWriter('./runs/' + generate_date + '/' + args.save_log)
    else:
        writer = None

    model_dir = path
    start_epoch = 0
    
    if args.evaluate:
        load_state_ckpt(args.checkpoint_path, model)
    else:
        best_prec1, start_epoch = load_state(model_dir, model, optimizer=optimizer)

    cudnn.benchmark = True
    cudnn.enabled = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImagenetDataset(
        args.train_root,
        args.train_source,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_dataset_wo_ms = ImagenetDataset(
        args.train_root,
        args.train_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = ImagenetDataset(
        args.val_root,
        args.val_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size//args.world_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    train_loader_wo_ms = DataLoader(
        train_dataset_wo_ms, batch_size=args.batch_size//args.world_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset, batch_size=50, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, writer, logging)
        return

    niters = len(train_loader)

    lr_scheduler = LRScheduler(optimizer, niters, args)

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        
        if args.early_fix_arch:
            if len(model.fix_arch_index.keys()) > 0:
                for key, value_lst in model.fix_arch_index.items():
                    model.log_alpha.data[key, :] = value_lst[1]
            sort_log_alpha = torch.topk(F.softmax(model.log_alpha.data, dim=-1), 2)
            argmax_index = (sort_log_alpha[0][:,0] - sort_log_alpha[0][:,1] >= 0.3)
            for id in range(argmax_index.size(0)):
                if argmax_index[id] == 1 and id not in model.fix_arch_index.keys():
                    model.fix_arch_index[id] = [sort_log_alpha[1][id,0].item(), model.log_alpha.detach().clone()[id, :]]
            
        if args.rank == 0 and args.SinglePath:
            logging.info('epoch %d', epoch)
            logging.info(model.log_alpha)         
            logging.info(F.softmax(model.log_alpha, dim=-1))         
            logging.info('flops %fM', model.cal_flops())  

        # train for one epoch
        if epoch >= args.epochs - 5 and args.lr_mode == 'step' and args.off_ms:
            train(train_loader_wo_ms, model, criterion, optimizer, arch_optimizer, lr_scheduler, epoch, writer, logging)
        else:
            train(train_loader, model, criterion, optimizer, arch_optimizer, lr_scheduler, epoch, writer, logging)


        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer, logging)
        if args.gen_max_child:
            args.gen_max_child_flag = True
            prec1 = validate(val_loader, model, criterion, epoch, writer, logging)        
            args.gen_max_child_flag = False

        if rank == 0:
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint(model_dir, {
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

def train(train_loader, model, criterion, optimizer, arch_optimizer, lr_scheduler, epoch, writer, logging):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    world_size = args.world_size
    rank = args.rank

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr_scheduler.update(i, epoch)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)

        if args.SinglePath:
            output, error_loss, loss_alpha = model(input_var, target_var, criterion)
            loss = criterion(output, target_var) / world_size
        else:
            output = model(input_var)
            loss = criterion(output, target_var) / world_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        reduced_loss = loss.data.clone()
        reduced_prec1 = prec1.clone() / world_size
        reduced_prec5 = prec5.clone() / world_size

        dist.all_reduce_multigpu([reduced_loss])
        dist.all_reduce_multigpu([reduced_prec1])
        dist.all_reduce_multigpu([reduced_prec5])

        losses.update(reduced_loss.item(), input.size(0))
        top1.update(reduced_prec1.item(), input.size(0))
        top5.update(reduced_prec5.item(), input.size(0))

        # compute gradient and do SGD step
        if args.random_sample:
            loss.backward()        
        # loss.backward()
        average_gradients(model)
        optimizer.step()
        optimizer.zero_grad()

        if args.SinglePath:
            if not args.random_sample and epoch >= args.pretrain_epoch:
                 arch_optimizer.step()
        arch_optimizer.zero_grad()

        if not args.random_sample:
            add_tensorboard_data(epoch, i, model, writer, train_loader)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and rank == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            niter = epoch * len(train_loader) + i
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], niter)
            writer.add_scalar('Train/Avg_Loss', losses.avg, niter)
            writer.add_scalar('Train/Avg_Top1', top1.avg / 100.0, niter)
            writer.add_scalar('Train/Avg_Top5', top5.avg / 100.0, niter)
            logging.info('train %03d %e %f %f', i, losses.avg, top1.avg, top5.avg)
    if rank == 0:
        logging.info('train %03d %e %f %f', i, losses.avg, top1.avg, top5.avg)

def validate(val_loader, model, criterion, epoch, writer, logging):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    world_size = args.world_size
    rank = args.rank

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target)

            # compute output
            if args.SinglePath:
                output, error_loss, loss_alpha = model(input_var)
                loss = criterion(output, target_var) / world_size     
       
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data.clone()
            reduced_prec1 = prec1.clone() / world_size
            reduced_prec5 = prec5.clone() / world_size

            dist.all_reduce_multigpu([reduced_loss])
            dist.all_reduce_multigpu([reduced_prec1])
            dist.all_reduce_multigpu([reduced_prec5])

            losses.update(reduced_loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and rank == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        if rank == 0:
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            if not args.gen_max_child_flag:
                niter = (epoch + 1)
                writer.add_scalar('Eval/Avg_Loss', losses.avg, niter)
                writer.add_scalar('Eval/Avg_Top1', top1.avg / 100.0, niter)
                writer.add_scalar('Eval/Avg_Top5', top5.avg / 100.0, niter)
                logging.info('valid %e %f %f', losses.avg, top1.avg, top5.avg)
            else:
                #derived child network with torch.argmax(model.log_alpha, dim=1)
                niter = (epoch + 1)
                writer.add_scalar('Eval_child/Avg_Top1', top1.avg / 100.0, niter)
                logging.info('child valid %e %f %f', losses.avg, top1.avg, top5.avg)
        
    return top1.avg

def add_tensorboard_data(epoch, step, model, writer, train_queue):
    if args.rank == 0:
        block_entropy = model._arch_entropy(model.log_alpha)
        
        cate_prob_next = F.softmax(model.log_alpha, dim=-1).detach()
        model.block_KL = model.KL(model.cate_prob, cate_prob_next)

        #iter train acc
        writer.add_scalar("total_entropy", block_entropy.sum(), step + len(train_queue.dataset) * epoch)
           
        #reward_block
        writer.add_scalar("block_reward_0", model.block_reward[0], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_1", model.block_reward[1], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_2", model.block_reward[2], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_3", model.block_reward[3], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_4", model.block_reward[4], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_5", model.block_reward[5], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_6", model.block_reward[6], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_7", model.block_reward[7], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_8", model.block_reward[8], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_9", model.block_reward[9], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_10", model.block_reward[10], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_11", model.block_reward[11], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_12", model.block_reward[12], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_13", model.block_reward[13], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_14", model.block_reward[14], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_15", model.block_reward[15], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_16", model.block_reward[16], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_17", model.block_reward[17], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_18", model.block_reward[18], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_reward_19", model.block_reward[19], step + len(train_queue.dataset) * epoch)
 
        #entropy_normal_edge
        writer.add_scalar("block_entropy_0", block_entropy[0], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_1", block_entropy[1], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_2", block_entropy[2], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_3", block_entropy[3], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_4", block_entropy[4], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_5", block_entropy[5], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_6", block_entropy[6], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_7", block_entropy[7], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_8", block_entropy[8], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_9", block_entropy[9], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_10", block_entropy[10], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_11", block_entropy[11], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_12", block_entropy[12], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_13", block_entropy[13], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_14", block_entropy[14], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_15", block_entropy[15], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_16", block_entropy[16], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_17", block_entropy[17], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_18", block_entropy[18], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_entropy_19", block_entropy[19], step + len(train_queue.dataset) * epoch)
      
        #KL_normal_edge
        writer.add_scalar("block_KL_0", model.block_KL[0], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_1", model.block_KL[1], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_2", model.block_KL[2], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_3", model.block_KL[3], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_4", model.block_KL[4], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_5", model.block_KL[5], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_6", model.block_KL[6], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_7", model.block_KL[7], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_8", model.block_KL[8], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_9", model.block_KL[9], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_10", model.block_KL[10], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_11", model.block_KL[11], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_12", model.block_KL[12], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_13", model.block_KL[13], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_14", model.block_KL[14], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_15", model.block_KL[15], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_16", model.block_KL[16], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_17", model.block_KL[17], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_18", model.block_KL[18], step + len(train_queue.dataset) * epoch)
        writer.add_scalar("block_KL_19", model.block_KL[19], step + len(train_queue.dataset) * epoch)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
