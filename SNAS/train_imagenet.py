import os
import sys
import numpy as np
import time
from datetime import datetime
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import dist_util
from torch.utils.data.distributed import DistributedSampler
from dataset import get_dataset

from model import NetworkImageNet as Network

import tensorboardX

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='/mnt/lustre/share/images', help='location of the data corpus')
#parser.add_argument('--data', type=str, default='/mnt/lustre/share/imagenet1k', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--warm_up_learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=50, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='SNAS_moderate', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--port', type=int, default=23333, help='port')
parser.add_argument('--remark', type=str, default='none', help='experiment name')
parser.add_argument('--warm_start', action='store_true', default=False, help='use warm start')
parser.add_argument('--warm_start_lr', type=float, default=0.001, help='init warm start learning rate')
parser.add_argument('--warm_start_gamma', type=float, default=1.2589254117941673, help='warm start learning rate mul')

args = parser.parse_args()

args.save = 'eval-{}-{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), args.arch, args.remark)
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#    format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)

CLASSES = 1000


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    rank, world_size = dist_util.dist_init(args.port, 'nccl')

    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    # logging.info('gpu device = %d' % args.gpu)
    if rank == 0:
        generate_date = str(datetime.now().date())
        utils.create_exp_dir(generate_date, args.save, scripts_to_save=glob.glob('*.py'))
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        logging.info("args = %s", args)
        logger = tensorboardX.SummaryWriter('./runs/eval_imagenet_{}_{}'.format(args.arch, args.remark))

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    if rank == 0:
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    if args.warm_start:
        lr = args.warm_start_lr / args.warm_start_gamma
    else:
        lr = args.learning_rate

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = get_dataset(traindir, os.path.join(args.data, 'meta/train.txt'), train_transform)

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    valid_dataset = get_dataset(validdir, os.path.join(args.data, 'meta/val.txt'), valid_transform)

    # train_queue = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size // world_size, sampler=DistributedSampler(train_dataset),
    #     pin_memory=True, num_workers=4)

    # valid_queue = torch.utils.data.DataLoader(
    #     valid_dataset, batch_size=args.batch_size // world_size, sampler=DistributedSampler(valid_dataset),
    #     pin_memory=True, num_workers=4)

    train_queue = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)


    if args.warm_start:
        scheduler = utils.WarmStart(optimizer, gamma=args.warm_start_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)

    best_acc_top1 = 0
    for epoch in range(args.epochs):
        '''if epoch == 0 or epoch == 1:
          for param_group in optimizer.param_groups:
            param_group['lr'] = args.warm_up_learning_rate
        elif epoch == 2:
          for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
        else:
          scheduler.step()'''

        scheduler.step()

        if rank == 0:
            logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer, rank)
        if rank == 0:
            logging.info('train_acc %f', train_acc)
            logger.add_scalar("epoch_train_acc", train_acc, epoch)
            logger.add_scalar("epoch_train_loss", train_obj, epoch)

        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion, rank)
        if rank == 0:
            logging.info('valid_acc_top1 %f', valid_acc_top1)
            logging.info('valid_acc_top5 %f', valid_acc_top5)
            logger.add_scalar("epoch_valid_acc_top1", valid_acc_top1, epoch)
            logger.add_scalar("epoch_valid_acc_top5", valid_acc_top5, epoch)

        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True

        if args.warm_start:
            # if not is_best and not scheduler.lr_const:
            if True:
                if rank == 0:
                    logging.info('warm start ended lr %e', scheduler.get_lr()[0])
                    logging.info("=> loading checkpoint '{}'".format(args.save))

                # checkpoint = torch.load(os.path.join(args.save, 'model_best.pth.tar'))

                checkpoint = torch.load(os.path.join(args.save, 'model_best.pth.tar'),
                                        map_location=lambda storage, loc: storage)

                best_acc_top1 = checkpoint['best_acc_top1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.lr_const = True
                if rank == 0:
                    logging.info('return to last checkpoint')

                del checkpoint  # dereference seems crucial
                torch.cuda.empty_cache()

                # args.start_epoch = checkpoint['epoch']
                '''best_acc_top1 = checkpoint['best_acc_top1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.lr_const = True
                if rank == 0:
                  logging.info('return to last checkpoint')'''

        if rank == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save)


def train(train_queue, model, criterion, optimizer, rank):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda()
        input = input.cuda()

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        # dist_util.sync_grad(model.parameters())
        optimizer.step()
        # dist_util.sync_bn_stat(model)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        loss = loss.detach()
        # dist_util.all_reduce([loss, prec1, prec5], 'mean')
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0 and rank == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

        # return top1.avg, objs.avg


def infer(valid_queue, model, criterion, rank):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)

            loss = loss.detach()
            # dist_util.all_reduce([loss, prec1, prec5], 'mean')
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0 and rank == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        return top1.avg, top5.avg, objs.avg
            # return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()
