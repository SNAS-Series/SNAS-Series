import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist
from blocks import Shufflenet, Shuffle_Xception

class ShuffleNetV2_OneShot(nn.Module):
    def __init__(self, input_size=224, n_class=1000, args=None, architecture=None, channels_scales=None):
        super(ShuffleNetV2_OneShot, self).__init__()

        assert input_size % 32 == 0
        assert architecture is not None and channels_scales is not None

        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        
        self.args = args
        self.bn_affine = args.bn_affine
        self.bn_eps = args.bn_eps
        self.num_blocks = 4
        self.device = torch.device("cuda")
        if args.flops_loss:
            self.flops = torch.Tensor([[13396992., 15805440., 19418112., 13146112.],
            [ 7325696.,  8931328., 11339776., 12343296.],
            [ 7325696.,  8931328., 11339776., 12343296.],
            [ 7325696.,  8931328., 11339776., 12343296.],
            [26304768., 28111104., 30820608., 20296192.],
            [10599680., 11603200., 13108480., 16746240.],
            [10599680., 11603200., 13108480., 16746240.],
            [10599680., 11603200., 13108480., 16746240.],
            [30670080., 31673600., 33178880., 21199360.],
            [10317440., 10819200., 11571840., 15899520.],
            [10317440., 10819200., 11571840., 15899520.],
            [10317440., 10819200., 11571840., 15899520.],
            [10317440., 10819200., 11571840., 15899520.],
            [10317440., 10819200., 11571840., 15899520.],
            [10317440., 10819200., 11571840., 15899520.],
            [10317440., 10819200., 11571840., 15899520.],
            [30387840., 30889600., 31642240., 20634880.],
            [10176320., 10427200., 10803520., 15476160.],
            [10176320., 10427200., 10803520., 15476160.],
            [10176320., 10427200., 10803520., 15476160.]]).cuda()/1000000

        self.log_alpha = torch.nn.Parameter(
                torch.zeros(sum(self.stage_repeats), self.num_blocks).normal_(self.args.loc_mean, self.args.loc_std).cuda().requires_grad_())

        self._arch_parameters = [self.log_alpha]
        self.weights = Variable(torch.zeros_like(self.log_alpha))
        if self.args.early_fix_arch:
            self.fix_arch_index = {}

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel, eps=self.bn_eps),
            nn.ReLU(inplace=True),
        )

        self.features = nn.ModuleList()
        #self.features = []
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                #blockIndex = architecture[archIndex]
                base_mid_channels = outp // 2
                mid_channels = int(base_mid_channels * channels_scales[archIndex])
                archIndex += 1

                blocks = nn.ModuleList()
                blocks.append(Shufflenet(inp, outp, mid_channels=mid_channels, ksize=3, stride=stride, bn_affine=self.bn_affine, bn_eps=self.bn_eps))
                blocks.append(Shufflenet(inp, outp, mid_channels=mid_channels, ksize=5, stride=stride, bn_affine=self.bn_affine, bn_eps=self.bn_eps))
                blocks.append(Shufflenet(inp, outp, mid_channels=mid_channels, ksize=7, stride=stride, bn_affine=self.bn_affine, bn_eps=self.bn_eps))
                blocks.append(Shuffle_Xception(inp, outp, mid_channels=mid_channels, stride=stride, bn_affine=self.bn_affine, bn_eps=self.bn_eps))
                
                input_channel = output_channel
                self.features += [blocks]

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1], eps=self.bn_eps),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x, target=None, criterion=None):

        error_loss = 0
        loss_alpha = 0
        flops_loss = 0
        total_flops = 0

        if self.args.gen_max_child_flag:
            self.weights = torch.zeros_like(self.log_alpha).scatter_(1, torch.argmax(self.log_alpha, dim = -1).view(-1,1), 1)
        else:
            self.weights = self._get_weights(self.log_alpha)

        if self.args.early_fix_arch:
            if len(self.fix_arch_index.keys()) > 0:
                for key, value_lst in self.fix_arch_index.items():
                    self.weights[key, :].zero_()
                    self.weights[key, value_lst[0]] = 1                

        if not self.args.random_sample and not self.args.gen_max_child_flag:
            cate_prob =  F.softmax(self.log_alpha, dim=-1)
            self.cate_prob = cate_prob.clone().detach()            
            loss_alpha = torch.log((self.weights * F.softmax(self.log_alpha, dim=-1)).sum(-1)).sum()/self.args.world_size
            self.weights.requires_grad_()        

        x = self.first_conv(x)
        for i, block in enumerate(self.features):
            pos = (self.weights[i,:] == 1).nonzero().item()
            x = self.features[i][pos](x) * self.weights[i, pos]
            if self.args.flops_loss and not self.args.gen_max_child_flag:
                flops_loss += cate_prob[i,pos]*self.flops[i,pos]   
                total_flops += self.flops[i,pos]         
        
        x = self.conv_last(x)
        x = self.globalpool(x)
        
        if self.args.use_dropout:
            x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)

        if not self.args.random_sample and self.training and not self.args.gen_max_child_flag:
            error_loss = criterion(x, target)/self.args.world_size
            self.weights.grad = torch.zeros_like(self.weights)
            #self.block_reward = torch.autograd.grad(error_loss, self.weights, retain_graph=True, allow_unused=True)
            (error_loss+loss_alpha).backward()           
            self.block_reward = self.weights.grad.data.sum(-1)
            self.log_alpha.grad.data.mul_(self.block_reward.view(-1,1))
            if self.args.flops_loss and total_flops >= 290:
                (self.args.flops_loss_coef*flops_loss).backward()

        if self.args.SinglePath:
            return x, error_loss, loss_alpha

    def cal_flops(self):
        total_flops = (3*3*3*16*112*112+640*1*1*1024*7*7+ (2*1024-1)*1000)/1000000
        weights = torch.zeros_like(self.log_alpha).scatter_(1, torch.argmax(self.log_alpha, dim = -1).view(-1,1), 1)
        for i in range(weights.size(0)):
            pos = (weights[i,:] == 1).nonzero().item()
            total_flops += self.flops[i,pos]
        return total_flops

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_weights(self, log_alpha):
        if self.args.random_sample:
            uni = torch.ones_like(log_alpha)
            m = torch.distributions.one_hot_categorical.OneHotCategorical(uni)
        else:
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=F.softmax(log_alpha, dim=-1))
        return m.sample()

    def arch_parameters(self):
        return self._arch_parameters

    def _arch_entropy(self, log_alpha):
        discrete_prob = F.softmax(log_alpha, dim=-1)
        epsilon = 1e-4
        discrete_prob = torch.max(discrete_prob, torch.ones_like(discrete_prob) * epsilon)
        arch_entropy = -(discrete_prob * torch.log(discrete_prob)).sum(-1)
        return arch_entropy

    def KL(self, p, q):
        """
           calculate KL(p||q)
        """
        return (p*torch.log(p/q)).sum(-1)


