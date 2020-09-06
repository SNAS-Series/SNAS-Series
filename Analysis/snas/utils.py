import os, sys
import numpy as np
import math
import torch
import shutil
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import logging
from datetime import datetime


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    device = torch.device("cuda")
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(device)
    return x/keep_prob*mask


def create_exp_dir(date, path, scripts_to_save=None):
    if not os.path.exists(date):
        os.mkdir(date)
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        script_path = os.path.join(path, 'scripts')
        if not os.path.exists(script_path):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.33, last_epoch=-1):
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def one_fifty_decay_100_072(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.last_epoch < 25:
            self.curr_temp = self.base_temp
        elif (self.last_epoch - 25) % 3 == 0 and self.last_epoch < 100:
            self.curr_temp = max(self.curr_temp * 0.9, self.temp_min)
        return self.curr_temp

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.total_epochs = 150
        self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        if self.curr_temp < self.temp_min:
            self.curr_temp = self.temp_min
        return self.curr_temp

    def decay_80_04(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.total_epochs = 150
        # self.curr_temp = (1-epoch/80)*(self.base_temp-0.4) + 0.4
        self.curr_temp = (1 - epoch / 40) * (self.base_temp - 0.33) + 0.33
        self.curr_temp = max(self.curr_temp, self.temp_min)
        return self.curr_temp

    def exp_decay(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.curr_temp = math.exp(-1.386 / 30 * self.last_epoch + 1.386)  # start from 4, epoch 30 decreases to 1
        self.curr_temp = max(self.curr_temp, self.temp_min)
        return self.curr_temp

    def stage_decay(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.last_epoch % 50 == 0:
            self.curr_temp = max(self.curr_temp * 0.6, self.temp_min)
        return self.curr_temp

    def periodic(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        period = 30
        self.temp_min = 0.003 * (3 ** (4 - self.last_epoch // period))
        self.curr_temp = (1 - (self.last_epoch % period) / period) * (self.base_temp - self.temp_min) + self.temp_min
        return self.curr_temp

    def periodic_max_min_decay(self, epoch=None):
        # 4 1
        # 2 0.67
        # 1 0.03
        # 4-1(5) 1-0.03(25)
        # 4-1(5) 1-0.03(25)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        period = 30
        if epoch < 30:
            self.curr_temp = 4 - (4 - 1) / period * epoch
        elif epoch >= 30 and epoch < 60:
            self.curr_temp = 2 - (2 - 0.67) / period * (epoch - 30)
        elif epoch >= 60 and epoch < 90:
            self.curr_temp = 1 - (1 - 0.33) / period * (epoch - 60)
        elif epoch >= 90:
            if epoch % period < 5:
                self.curr_temp = 4 - (4 - 0.33) / period * (epoch % 90)
            else:
                self.curr_temp = 0.33 - (0.33 - 0.03) / period * (epoch % 90)
        return self.curr_temp

    def cosine_annealing(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.curr_temp = self.temp_min + (self.base_temp - self.temp_min) * (
        1 + math.cos(math.pi * self.last_epoch / self.total_epochs)) / 2
        return self.curr_temp


def new_clip(arch_parameters, arch_parameters_ema, sigma):
    total_norm = 0
    count = 0
    # ratio = 0
    total_kt_g = 0
    for v, m in zip(arch_parameters, arch_parameters_ema):
        if v.grad is not None:
            kl = (torch.exp(v) / torch.exp(v).sum(-1, keepdim=True) * (v - m)).sum()
            k = torch.autograd.grad(kl, v, only_inputs=True)[0].data
            norm_2 = torch.norm(k, 2, 1, True) ** 2
            kt_g = (-v.grad * k).sum(-1, keepdim=True)
            clip = k * torch.max((kt_g - sigma) / norm_2, torch.zeros_like(v.grad))
            # ratio += (clip / v.grad).sum()
            total_kt_g += kt_g.sum()
            count += (clip.data.cpu().numpy()).size
            v.grad = v.grad + clip
            total_norm += norm_2.sum()
    return total_norm, total_kt_g / count


class Resource_Scheduler(object):
    def __init__(self, total_epochs, curr_epoch, curr_lambda, base_lambda, constant, valid_acc, valid_acc_pre,
                 valid_acc_pre_pre, lambda_pre, add_dummy, slope_flag, epoch_flag_add, epoch_flag_rm, ema_valid_acc, mavg_alpha):
        self.total_epochs = total_epochs
        self.curr_epoch = curr_epoch
        self.curr_lambda = curr_lambda
        self.base_lambda = base_lambda
        self.constant = constant
        self.valid_acc = valid_acc
        self.valid_acc_pre = valid_acc_pre
        self.valid_acc_pre_pre = valid_acc_pre_pre
        self.lambda_pre = lambda_pre
        self.add_dummy = add_dummy
        self.slope_flag = slope_flag
        self.epoch_flag_add = epoch_flag_add
        self.epoch_flag_rm = epoch_flag_rm
        self.ema_valid_acc = ema_valid_acc
        self.mavg_alpha = mavg_alpha
        self.ema_valid_acc_pre = self.valid_acc_pre

    def step(self, epoch = None):
        return self.acc_scheduler(self.lambda_constant())

    def ema(self):
        a = self.mavg_alpha
        if self.curr_epoch == 0:
            return self.valid_acc, self.valid_acc_pre
        else:
            return (1-a)*self.valid_acc_pre + a*self.valid_acc, (1-a)*self.valid_acc_pre_pre + a*self.valid_acc_pre

    def acc_scheduler(self, shape):
        self.ema_valid_acc, self.ema_valid_acc_pre = self.ema()
        slope = self.ema_valid_acc - self.ema_valid_acc_pre
        logging.info("slope: %f, original slope: %f, valid_acc: %f, valid_pre: %f, ema_valid_acc: %f, ema_valid_acc_pre: %f",
                     slope, self.valid_acc - self.valid_acc_pre, self.valid_acc, self.valid_acc_pre, self.ema_valid_acc,
                     self.ema_valid_acc_pre)
        if self.add_dummy == 0:
            if abs(slope) <= self.slope_flag or self.curr_epoch > self.epoch_flag_add:
                return shape, 1, self.ema_valid_acc
            else:
                return self.base_lambda, 0, self.ema_valid_acc
        else:
            if self.curr_epoch > self.epoch_flag_rm:
                return self.base_lambda, 0, self.ema_valid_acc
            else:
                return shape, 1, self.ema_valid_acc

    def lambda_constant(self):
        self.curr_lambda = self.constant
        return self.curr_lambda

    def lambda_linear(self, first = 0.0001):
        self.curr_lambda = self.curr_lambda + first
        return self.curr_lambda

    def lambda_poly(self, epoch=None, first = 0.0001, second = 0.0000002):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.curr_lambda = self.curr_lambda + first + 0.5 * second
        self.curr_lambda = self.ema()
        # self.curr_lambda = self.curr_lambda + first * self.last_epoch + 0.5 * second
        return self.curr_lambda

def txt(folder, training_name, txt_name, writein, generate_date):
    b = os.path.abspath('.') + '/' + folder + '/' + generate_date + '/' + training_name

    if not os.path.exists(b):
        os.makedirs(b)
    x = b + '/' + txt_name +'.txt'
    file = open(x, 'w')
    file.write(writein)
    file.close()

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def calc_grad(args, optimizer, arch_optimizer, error_loss, size, flops, mac, log_alpha, retain):
    optimizer.zero_grad()
    arch_optimizer.zero_grad()
    error_loss.backward(retain_graph=True)
    error_loss_grad = log_alpha.grad.cpu().numpy()

    optimizer.zero_grad()
    arch_optimizer.zero_grad()
    size.backward(retain_graph=True)
    size_grad = log_alpha.grad.cpu().numpy()

    optimizer.zero_grad()
    arch_optimizer.zero_grad()
    flops.backward(retain_graph=True)
    flops_grad = log_alpha.grad.cpu().numpy()

    optimizer.zero_grad()
    arch_optimizer.zero_grad()
    mac.backward(retain_graph=True)
    mac_grad = log_alpha.grad.cpu().numpy()

    n = error_loss_grad.shape[0] * error_loss_grad.shape[1]

    optimizer.zero_grad()
    arch_optimizer.zero_grad()
    if args.update_grad:
        error_loss.backward(retain_graph=retain)
    elif not args.update_grad and not args.step_length:
        error_loss.backward(retain_graph=retain)
    else:
        error_loss.backward(retain_graph=True)

    return np.concatenate((error_loss_grad.reshape((n, 1)), size_grad.reshape((n, 1)), flops_grad.reshape((n, 1)),
                     mac_grad.reshape((n, 1))), axis=1)

def qp_solver(logging, G=None, C=None, b=None):
    # solver quadratic programming in the form of
    # minimize 1/2x^tGx + a^tx st C^tx=b, x>=0

    G = 0.5 * (G.T + G)# guarantee symmetric
    qp_G = (G * 1e-10).astype('float64')

    objF_num = qp_G.shape[0]
    qp_a = np.zeros(objF_num).reshape((objF_num,))
    qp_C = C.T
    qp_b = b
    meq = 1 # first meq constraints satisfy equalities
    error_flag = False
    error_msg = None
    #lambda_opt = qp.solve_qp(G=qp_G, a=qp_a, C=qp_C, b=qp_b, meq=meq, factorized=False)[0]

    beg = datetime.now()
    try:
        lambda_opt = qp.solve_qp(G=qp_G, a=qp_a, C=qp_C, b=qp_b, meq=meq, factorized=False)[0]
    except Exception as e:
        error_flag = True
        lambda_opt = np.zeros((objF_num,))
        error_msg = e
        #logging.info('ERROR while trying to solve QP: %s ', str(e))
        #logging.info('set the optimal as zero')
    end = datetime.now()
    return lambda_opt, error_flag, (end-beg).total_seconds(), error_msg

def pareto_critical(args, optimizer, arch_optimizer, error_loss, size, flops, mac, log_alpha, retain=True):
    J = calc_grad(args, optimizer, arch_optimizer, error_loss, size, flops, mac, log_alpha, retain)

    # J*x = 0, for all x >= 0, or x is in Null(J) for all x >= 0
    ns = null_space(J)
    #basis_num = ns.shape[1]
    if np.all(ns >= 0): #TODO
        return True, (ns >= 0).astype(int)
    else:
        return False, (ns >= 0).astype(int)

def iteration(model, input, target, args, criterion, optimizer, arch_optimizer, error_loss, size, flops, mac, log_alpha,
              no_ite_critical_flag, logging, cell, retain=True):

    J = calc_grad(args, optimizer, arch_optimizer, error_loss, size, flops, mac, log_alpha, retain)
    G = np.dot(J.T, J)
    C = np.vstack([np.ones(J.shape[1]), np.identity(J.shape[1])])  # (5, 4)
    b = np.concatenate((np.ones(1), np.zeros(J.shape[1]))).reshape((J.shape[1] + 1,))
    lambda_opt, error_flag, solver_period, error_msg = qp_solver(logging, G, C, b)

    s_ = - np.dot(J, lambda_opt)
    s = (s_).reshape((model._k, model._num_ops))
    t_period, t_iteration, update_period = pareto_update(s, args, input, target, model, criterion, log_alpha, error_flag, no_ite_critical_flag, error_loss, size, flops, mac, J, cell)
    return t_period, t_iteration, solver_period, update_period, error_flag, error_msg, J, s_

def pareto_update(s, args, input, target, model, criterion, log_alpha, error_flag, no_ite_critical_flag, error_loss0, size0, flops0, mac0, J, cell):
    delta_log_alpha = (torch.from_numpy(s).type(log_alpha.grad.dtype)).cuda()
    t_period = 0
    t_iteration = 0
    if not args.update_grad: # set gradients of weights to zero, manually change weights directly
        if args.step_length:
            beg = datetime.now()
            t, t_iteration = step_length(input, target, args, model, criterion, log_alpha, s, delta_log_alpha, error_loss0, size0, flops0, mac0, J, cell)
            end = datetime.now()
            t_period = (end-beg).total_seconds()
        else:
            t = 1
        beg = datetime.now()
        log_alpha.data += delta_log_alpha * t

        # log_alpha.grad.data = None # Error NoneType object doesn't have attribute 'data'

        # TODO: if error occurs, whether update with SGD or no update -- to be determined
        if not error_flag:
            log_alpha.grad.data.zero_()

        if not args.pareto_iteration and no_ite_critical_flag:
            log_alpha.grad.data.zero_()

    else: # set gradients of weights to optimal delta weights
        beg = datetime.now()
        log_alpha.grad = (torch.from_numpy(s).type(log_alpha.grad.dtype)).cuda()
    end = datetime.now()
    update_period = (end - beg).total_seconds()
    return t_period, t_iteration, update_period

def criteria(input, target, args, model, criterion, error_loss0, size0, flops0, mac0, J, cell, t, s):
    logits, penalty, op_normal, op_reduce = model(input)
    error_loss = criterion(logits, target)

    if not args.log_penalty:
        size_normal = penalty[29]
        size_reduce = penalty[32]
        flops_normal = penalty[30]
        flops_reduce = penalty[32]
        mac_normal = penalty[31]
        mac_reduce = penalty[33]
    else:
        size_normal = penalty[41]
        size_reduce = penalty[44]
        flops_normal = penalty[42]
        flops_reduce = penalty[45]
        mac_normal = penalty[43]
        mac_reduce = penalty[46]

    crite = []
    beta = 9e-1
    update_error_loss = (torch.from_numpy(np.dot(J[:,0], s.reshape((model._k * model._num_ops,1)))).type(error_loss.dtype)).cuda()
    update_size = (torch.from_numpy(np.dot(J[:,1], s.reshape((model._k * model._num_ops,1)))).type(error_loss.dtype)).cuda()
    update_flops = (torch.from_numpy(np.dot(J[:,2], s.reshape((model._k * model._num_ops,1)))).type(error_loss.dtype)).cuda()
    update_mac = (torch.from_numpy(np.dot(J[:,3], s.reshape((model._k * model._num_ops,1)))).type(error_loss.dtype)).cuda()


    crite.append(error_loss <= error_loss0 + beta * t * update_error_loss)
    if cell == 'normal':
        crite.append(size_normal <= size0 + beta * t * update_size)
        crite.append(flops_normal <= flops0 + beta * t * update_flops)
        crite.append(mac_normal <= mac0 + beta * t * update_mac)
    else:
        crite.append(size_reduce <= size0 + beta * t * update_size)
        crite.append(flops_reduce <= flops0 + beta * t * update_flops)
        crite.append(mac_reduce <= mac0 + + beta * t * update_mac)

    for c in crite:
        if c == False:
            return False

    return True


def step_length(input, target, args, model, criterion, log_alpha, s, delta_log_alpha, error_loss0, size0,
                flops0, mac0, J, cell, t0=1, t_iteration=0):
    t = t0
    log_alpha.data += delta_log_alpha * t

    while (not criteria(input, target, args, model, criterion, error_loss0, size0, flops0, mac0, J, cell, t, s)) and (t > args.step_length_limit):
        t_iteration += 1
        t /= 2
        step_length(input, target, args, model, criterion, log_alpha, s, delta_log_alpha, error_loss0, size0,
                flops0, mac0, J, cell, t0=t, t_iteration=t_iteration)

    return t, t_iteration

