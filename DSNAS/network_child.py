import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist
from blocks import Shufflenet, Shuffle_Xception

class ShuffleNetV2_OneShot(nn.Module):
    def __init__(self, input_size=224, n_class=1000, args=None, architecture=None, channels_scales=None, weights=None):
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

        self.log_alpha = torch.nn.Parameter(
                torch.zeros(sum(self.stage_repeats), self.num_blocks).normal_(self.args.loc_mean, self.args.loc_std).cuda().requires_grad_())

        self._arch_parameters = [self.log_alpha]
        #self.weights = Variable(torch.zeros_like(self.log_alpha))
        self.weights = weights

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel, eps=self.bn_eps),
            nn.ReLU(inplace=True),
        )

        self.features = nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                base_mid_channels = outp // 2
                mid_channels = int(base_mid_channels * channels_scales[archIndex])
                pos = (self.weights[archIndex,:] == 1).nonzero().item()
                archIndex += 1

                blocks = nn.ModuleList()
                if pos == 0:
                    blocks.append(Shufflenet(inp, outp, mid_channels=mid_channels, ksize=3, stride=stride, bn_affine=self.bn_affine, bn_eps=self.bn_eps))
                    blocks.append(None)
                    blocks.append(None)
                    blocks.append(None)
                elif pos == 1:
                    blocks.append(None)
                    blocks.append(Shufflenet(inp, outp, mid_channels=mid_channels, ksize=5, stride=stride, bn_affine=self.bn_affine, bn_eps=self.bn_eps))
                    blocks.append(None)
                    blocks.append(None)
                elif pos == 2:
                    blocks.append(None)
                    blocks.append(None)
                    blocks.append(Shufflenet(inp, outp, mid_channels=mid_channels, ksize=7, stride=stride, bn_affine=self.bn_affine, bn_eps=self.bn_eps))
                    blocks.append(None)
                elif pos == 3:
                    blocks.append(None)
                    blocks.append(None)
                    blocks.append(None)
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

    def _reset_bn_running_stats(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                if self.args.reset_bn_affine:
                    m.reset_parameters()
                else:
                    m.reset_running_stats()
    
    def forward(self, x, target=None, criterion=None):

        error_loss = 0
        loss_alpha = 0

        x = self.first_conv(x)
        for i, block in enumerate(self.features):
            pos = (self.weights[i,:] == 1).nonzero().item()
            x = self.features[i][pos](x) * self.weights[i, pos]
#        x = self.features(x)
        
        x = self.conv_last(x)
        x = self.globalpool(x)
        
        x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)

        return x
    
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

