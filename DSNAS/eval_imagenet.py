from __future__ import division
import argparse
import os
import glob
import time
from datetime import datetime
import torch.distributed as dist
import torch
import utils
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import yaml
import sys
from tensorboardX import SummaryWriter
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))

from devkit.core import (init_dist, broadcast_params, average_gradients, load_state_ckpt, load_state, save_checkpoint, LRScheduler, CrossEntropyLoss)
from devkit.dataset.imagenet_dataset import ImagenetDataset
from network_eval import ShuffleNetV2_OneShot

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
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--off-ms', action='store_true')

parser.add_argument('--epochs', type=int, default=240, help='num of training epochs')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--loc_mean', type=float, default=1, help='initial mean value to generate the location')
parser.add_argument('--loc_std', type=float, default=0.01, help='initial std to generate the location')
parser.add_argument('--bn_affine', action='store_true', default=False, help='bn affine flag')
parser.add_argument('--bn_eps', type=float, default=1e-2, help='initial mean value to generate the location')

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

    # create model
    print("=> creating model '{}'".format(args.model))
    if args.SinglePath:
        architecture = args.arch
        scale_list = 8*[1.0]
        scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        channels_scales = []
        for i in range(len(scale_ids)):
            channels_scales.append(scale_list[scale_ids[i]])
        model = ShuffleNetV2_OneShot(args=args, architecture=architecture, channels_scales=channels_scales)
        model.cuda()
        broadcast_params(model)

    # auto resume from a checkpoint
    remark = 'imagenet_'

    if args.remark != 'none':
        remark += args.remark

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

    #model_dir = args.model_dir
    model_dir = path
    start_epoch = 0

    if args.evaluate:
        load_state_ckpt(args.checkpoint_path, model)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = ImagenetDataset(
        args.val_root,
        args.val_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_sampler = DistributedSampler(val_dataset)

    val_loader = DataLoader(
        val_dataset, batch_size=50, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, 0, writer, logging)
        return

def validate(val_loader, model, epoch, writer, logging):
    batch_time = AverageMeter()
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

            output = model(input_var)

            # measure accuracy
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            reduced_prec1 = prec1.clone() / world_size
            reduced_prec5 = prec5.clone() / world_size

            dist.all_reduce_multigpu([reduced_prec1])
            dist.all_reduce_multigpu([reduced_prec5])

            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and rank == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    top1=top1, top5=top5))

        if rank == 0:
            niter = (epoch + 1)
            logging.info('valid %f %f', top1.avg, top5.avg)
        
    return top1.avg


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
