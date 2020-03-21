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

