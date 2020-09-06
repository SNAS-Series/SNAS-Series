import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search_cost import Network
from architect_cost import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--cal_stat', action='store_true', default=False, help='cal stat')
parser.add_argument('--resume', action='store_true', default=False, help='resume flag')
parser.add_argument('--resume_path', type=str, help='resume path')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--steps', type=int, default=4, help='num of init steps')
parser.add_argument('--multiplier', type=int, default=4, help='num of multipliers')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--update_alpha', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--remark', type=str, default='none', help='location of the data corpus')

parser.add_argument('--fix_edge0', action='store_true', default=False, help='fix edge 0')
parser.add_argument('--fix_edge1', action='store_true', default=False, help='fix edge 1')
parser.add_argument('--fix_edge2', action='store_true', default=False, help='fix edge 2')
parser.add_argument('--fix_edge3', action='store_true', default=False, help='fix edge 3')
parser.add_argument('--fix_edge4_noskip', action='store_true', default=False, help='fix edge 3')

parser.add_argument('--del_edge0', action='store_true', default=False, help='del edge 0')
parser.add_argument('--del_edge1', action='store_true', default=False, help='del edge 1')
parser.add_argument('--del_edge2', action='store_true', default=False, help='del edge 2')
parser.add_argument('--del_edge3', action='store_true', default=False, help='del edge 3')
parser.add_argument('--del_edge4', action='store_true', default=False, help='del edge 4')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
if args.remark != 'none':
    args.save = args.save + '_layer_' + str(args.layers) + '_epoch_' + str(args.epochs)  + '_steps_' + str(args.steps) + '_multi_' + str(args.multiplier) + '_'  + args.remark
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
normal_reward_mean = 0
normal_reward_mean_square = 0
reduce_reward_mean = 0
reduce_reward_mean_square = 0
count = 0


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args, args.init_channels, CIFAR_CLASSES, args.layers, criterion, args.steps, args.multiplier)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  
  if args.resume:
    model.load_state_dict(torch.load(args.resume_path)) 
 
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=0)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=0)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info(model.alphas_normal)
    logging.info(model.alphas_reduce)    
    logging.info(F.softmax(model.alphas_normal, dim=-1))
    logging.info(F.softmax(model.alphas_reduce, dim=-1))
    logging.info('genotype = %s', genotype)
    
    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    if args.cal_stat:
        logging.info('normal reward mean')
        logging.info(normal_reward_mean)
        logging.info('normal reward variance')
        logging.info(-normal_reward_mean**2+normal_reward_mean_square)
        logging.info('reduce reward mean')
        logging.info(reduce_reward_mean)
        logging.info('reduce reward variance')
        logging.info(-reduce_reward_mean**2+reduce_reward_mean_square)
        logging.info('normal reward total mean')
        logging.info(normal_reward_mean.sum(0))

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top2 = utils.AvgrageMeter()
  top3 = utils.AvgrageMeter()
  top4 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  global normal_reward_mean
  global normal_reward_mean_square
  global reduce_reward_mean
  global reduce_reward_mean_square
  global count

  train_correct_count = 0
  train_correct_cost = 0
  train_correct_entropy = 0
  train_correct_loss = 0
  train_wrong_count = 0
  train_wrong_cost = 0
  train_wrong_entropy = 0
  train_wrong_loss = 0
  search_correct_count = 0
  search_correct_cost = 0
  search_correct_entropy = 0
  search_correct_loss = 0
  search_wrong_count = 0
  search_wrong_cost = 0
  search_wrong_entropy = 0
  search_wrong_loss = 0

  normal_reward_mean = 0
  normal_reward_mean_square = 0
  reduce_reward_mean = 0
  reduce_reward_mean_square = 0
  count = 0

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)
    #logits = model(input)
    #loss = criterion(logits, target)

    #loss.backward()


    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search).cuda()
    target_search = Variable(target_search).cuda(async=True)
    logits, loss = architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
    for i in range(logits.size(0)):

        index = logits[i].topk(5,0,True, True)[1]
        if  index[0].item() == target_search[i].item():
            search_correct_cost += (-logits[i, target_search[i].item()] + (F.softmax(logits[i])*logits[i]).sum())
            search_correct_count += 1
            discrete_prob = F.softmax(logits[i], dim=-1)
            search_correct_entropy += -(discrete_prob * torch.log(discrete_prob)).sum(-1)
            search_correct_loss += -torch.log(discrete_prob)[target_search[i].item()]
        else:
            search_wrong_cost += (-logits[i, target_search[i].item()] + (F.softmax(logits[i])*logits[i]).sum())
            search_wrong_count += 1
            discrete_prob = F.softmax(logits[i], dim=-1)
            search_wrong_entropy += -(discrete_prob * torch.log(discrete_prob)).sum(-1)
            search_wrong_loss += -torch.log(discrete_prob)[target_search[i].item()]
#
    if args.cal_stat:
        count += 1
        normal_reward_mean = (normal_reward_mean*(count-1)+model.weights_normal.grad)/count
        normal_reward_mean_square = (normal_reward_mean_square*(count-1)+model.weights_normal.grad**2)/count
        reduce_reward_mean = (reduce_reward_mean*(count-1)+model.weights_reduce.grad)/count
        reduce_reward_mean_square = (reduce_reward_mean_square*(count-1)+model.weights_reduce.grad**2)/count
        model.weights_normal.grad.zero_()
        model.weights_reduce.grad.zero_()
    
    optimizer.zero_grad()
    #loss = criterion(logits, target_search.cuda())
    logits = model(input)
    loss = criterion(logits, target)

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

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    #prec1, prec2, prec3, prec4, prec5 = utils.accuracy(logits, target_search, topk=(1, 2, 3, 4, 5))   
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))   
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  logging.info('train correct loss ')
  logging.info((train_correct_loss/train_correct_count).item())
  logging.info('train correct entropy ')
  logging.info((train_correct_entropy/train_correct_count).item())
  logging.info('train correct cost ')
  logging.info((train_correct_cost/train_correct_count).item())
  logging.info('train correct count ')
  logging.info(train_correct_count)

  logging.info('train wrong loss ')
  logging.info((train_wrong_loss/train_wrong_count).item())
  logging.info('train wrong entropy ')
  logging.info((train_wrong_entropy/train_wrong_count).item())
  logging.info('train wrong cost ')
  logging.info((train_wrong_cost/train_wrong_count).item())
  logging.info('train wrong count ')
  logging.info(train_wrong_count)

  logging.info('train total loss ')
  logging.info(((train_correct_loss+train_wrong_loss)/(train_correct_count+train_wrong_count)).item())
  logging.info('train total entropy ')
  logging.info(((train_correct_entropy+train_wrong_entropy)/(train_correct_count+train_wrong_count)).item())
  logging.info('train total cost ')
  logging.info(((train_correct_cost+train_wrong_cost)/(train_correct_count+train_wrong_count)).item())
  logging.info('train total count ')
  logging.info(train_correct_count+train_wrong_count)

  logging.info('search correct loss ')
  logging.info((search_correct_loss/search_correct_count).item())
  logging.info('search correct entropy ')
  logging.info((search_correct_entropy/search_correct_count).item())
  logging.info('search correct cost ')
  logging.info((search_correct_cost/search_correct_count).item())
  logging.info('search correct count ')
  logging.info(search_correct_count)

  logging.info('search wrong loss ')
  logging.info((search_wrong_loss/search_wrong_count).item())
  logging.info('search wrong entropy ')
  logging.info((search_wrong_entropy/search_wrong_count).item())
  logging.info('search wrong cost ')
  logging.info((search_wrong_cost/search_wrong_count).item())
  logging.info('search wrong count ')
  logging.info(search_wrong_count)

  logging.info('search total loss ')
  logging.info(((search_correct_loss+search_wrong_loss)/(search_correct_count+search_wrong_count)).item())
  logging.info('search total entropy ')
  logging.info(((search_correct_entropy+search_wrong_entropy)/(search_correct_count+search_wrong_count)).item())
  logging.info('search total cost ')
  logging.info(((search_correct_cost+search_wrong_cost)/(search_correct_count+search_wrong_count)).item())
  logging.info('search total count ')
  logging.info(search_correct_count+search_wrong_count)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  #total_count = 0
  #acc_count = 0
  #cost = 0
  with torch.no_grad():
      for step, (input, target) in enumerate(valid_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)
        #input = Variable(input, volatile=True).cuda()
        #target = Variable(target, volatile=True).cuda(async=True)

        logits = model(input)
        loss = criterion(logits, target)

        #total_count += logits.size(0)
        #for i in range(logits.size(0)):
        #    if F.softmax(logits)[i, target[i].item()] > 0.5:
        #        acc_count += 1
        #    cost += (-logits[i, target[i].item()] + (F.softmax(logits[i])*logits[i]).sum())

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        # objs.update(loss.data[0], n)
        # top1.update(prec1.data[0], n)
        # top5.update(prec5.data[0], n)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  #print(cost/total_count)
  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

