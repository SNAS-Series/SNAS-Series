import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights, j, fix_edge4_noskip):
      result = 0
      count = 0
      for weight, op in zip(weights, self._ops):
          if not (fix_edge4_noskip and count == 3 and j==4):
            result += weight * op(x)
          count += 1
      return result
    #return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights, fix_edge4_noskip):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    count = 0
    for i in range(self._steps):
      #s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      s = 0
      for j, h in enumerate(states):
        s += self._ops[offset+j](h, weights[offset+j], count, fix_edge4_noskip)
        count += 1
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, args, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self.args = args
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)
    self.weights_normal = Variable(torch.zeros(layers, k, num_ops).cuda(), requires_grad=True)
    self.weights_reduce = Variable(torch.zeros(layers, k, num_ops).cuda(), requires_grad=True)
    self.weights_normal.grad = torch.zeros_like(self.weights_normal)
    self.weights_reduce.grad = torch.zeros_like(self.weights_reduce)
    self.correct_cost = 0
    self.correct_count = 0
    self.correct_entropy = 0
    self.wrong_cost = 0
    self.wrong_count = 0
    self.wrong_entropy = 0

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
    #   if i == 0:
    #     C_curr *= 2
    #     reduction = True
    #   else:
    #     reduction = False
    # for i in range(layers):
    #   if i in [layers//3, 2*layers//3] and i != 0:
    #     C_curr *= 2
    #     reduction = True
    #   else:
    #     reduction = False
      reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes, bias=False)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    self.weights_normal.grad = torch.zeros_like(self.weights_normal)
    self.weights_reduce.grad = torch.zeros_like(self.weights_reduce)
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        if self.args.cal_stat:
            self.weights_reduce.data[i,:,:] = F.softmax(self.alphas_reduce, dim=-1)
            if self.args.del_edge3:
              self.weights_reduce[3,:].data.zero_()

            if self.args.del_edge1:
              self.weights_reduce[1,:].data.zero_()  

            if self.args.del_edge0:
              self.weights_reduce[0,:].data.zero_()

            if self.args.del_edge2:
              self.weights_reduce[2,:].data.zero_()  

            if self.args.fix_edge0:
              self.weights_reduce[0,:].data.zero_()
              self.weights_reduce.data[0,7] = 1

            if self.args.fix_edge1:
              self.weights_reduce[1,:].data.zero_()  

            if self.args.fix_edge2:
              self.weights_reduce[2,:].data.zero_()

            if self.args.fix_edge3:
              self.weights_reduce[3,:].data.zero_()  
            s0, s1 = s1, cell(s0, s1, self.weights_reduce)
        else:
            weights = F.softmax(self.alphas_reduce, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
      else:
        #self.weights.data = self._get_onehot_mask(self.alphas_normal)
        if self.args.cal_stat:
            for j in range(self.weights_normal.size(1)):
                if j ==4 and self.args.fix_edge4_noskip:
                    self.weights_normal.data[i,j,[0,1,2,4,5,6,7]] = F.softmax(self.alphas_normal[j,[0,1,2,4,5,6,7]], dim=-1)
                else:
                    self.weights_normal.data[i,j,:] = F.softmax(self.alphas_normal[j], dim=-1)

            if self.args.del_edge3:
              self.weights_normal.data[i,3,:].zero_()
              self.weights_normal.data[i,3,0] = 1

            if self.args.del_edge1:
              self.weights_normal.data[i,1,:].zero_()  
              self.weights_normal.data[i,1,0] = 1

            if self.args.del_edge0:
              self.weights_normal.data[i,0,:].zero_()
              self.weights_normal.data[i,0,0] = 1

            if self.args.del_edge2:
              self.weights_normal.data[i,2,:].zero_()  
              self.weights_normal.data[i,2,0] = 1

            if self.args.del_edge4:
              self.weights_normal.data[i,4,:].zero_()  
              self.weights_normal.data[i,4,0] = 1

            if self.args.fix_edge0:
              self.weights_normal[i,0,:].data.zero_()
              self.weights_normal.data[i,0,7] = 1

            if self.args.fix_edge1:
              self.weights_normal[i,1,:].data.zero_()  

            if self.args.fix_edge2:
              self.weights_normal[i,2,:].data.zero_()

            if self.args.fix_edge3:
              self.weights_normal[i,3,:].data.zero_()  

            #if self.args.fix_edge4_noskip:
            #    self.weights_normal[i, 4,:].zero_()
            #    self.weights_normal.data[i, 4,[0,2,3,4,5,6,7]] = F.softmax(self.alphas_normal[4,[0,2,3,4,5,6,7]],dim=-1)

            s0, s1 = s1, cell(s0, s1, self.weights_normal[i], self.args.fix_edge4_noskip)
        else:
            weights = F.softmax(self.alphas_normal, dim=-1)
            #weights=Variable(torch.zeros_like(self.alphas_normal).cuda(), requires_grad=True)
            #weights.grad = torch.zeros_like(weights)
            #for j in range(weights.size(0)):
            #    if j ==4 and self.args.fix_edge4_noskip:
            #        weights[i,j,[0,1,2,4,5,6,7]] = F.softmax(self.alphas_normal[j,[0,1,2,4,5,6,7]], dim=-1)
            #    else:
            #        weights[i,j,:] = F.softmax(self.alphas_normal[j], dim=-1)

            if self.args.del_edge0:
                weights[0,:].data.zero_()

            if self.args.del_edge2:
                weights[2,:].data.zero_()  

            #if self.args.fix_edge4_noskip:
            #    weights[4,:].zero_()
            #    weights.data[4,[0,2,3,4,5,6,7]] = F.softmax(self.alphas_normal[4,[0,2,3,4,5,6,7]],dim=-1)

            s0, s1 = s1, cell(s0, s1, weights, self.args.fix_edge4_noskip)
      #if cell.reduction:
      #  weights = F.softmax(self.alphas_reduce, dim=-1) 
      #else:
      #  weights = F.softmax(self.alphas_normal, dim=-1)

      #if self.args.del_edge3:
      #  weights[3,:].data.zero_()

      #if self.args.del_edge1:
      #  weights[1,:].data.zero_()  

      #if self.args.del_edge0:
      #  weights[0,:].data.zero_()

      #if self.args.del_edge2:
      #  weights[2,:].data.zero_()  

      #if self.args.fix_edge0:
      #  weights[0,:].data.zero_()
      #  weights.data[0,7] = 1

      #if self.args.fix_edge1:
      #  weights[1,:].data.zero_()  

      #if self.args.fix_edge2:
      #  weights[2,:].data.zero_()

      #if self.args.fix_edge3:
      #  weights[3,:].data.zero_()  

      #s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

    #if step % args.report_freq == 0:
    #  logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  def _loss(self, input, target):
    logits = self(input)
    ###
    #self.weights_cost = Variable(torch.ones(1).cuda(), requires_grad=True)
    for i in range(logits.size(0)):
        if torch.argmax(logits[i]).item() == target[i].item():
            self.correct_cost += (-logits[i, target[i].item()] + (F.softmax(logits[i])*logits[i]).sum())
            self.correct_count += 1
            discrete_prob = F.softmax(logits[i], dim=-1)
            self.correct_entropy += -(discrete_prob * torch.log(discrete_prob)).sum(-1)
        else:
            self.wrong_cost += (-logits[i, target[i].item()] + (F.softmax(logits[i])*logits[i]).sum())
            self.wrong_count += 1
            discrete_prob = F.softmax(logits[i], dim=-1)
            self.wrong_entropy += -(discrete_prob * torch.log(discrete_prob)).sum(-1)
    ###
    #print(cost/logits.size(0))
    return logits, self._criterion(logits, target)

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)
    
    if not self.args.cal_stat:
        self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    else:
        self.alphas_normal = Variable(1e-3*torch.ones(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3*torch.ones(k, num_ops).cuda(), requires_grad=True)
        
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) for k in range(len(W[x]))))[:] #if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            #if k != PRIMITIVES.index('none'):
            #  if k_best is None or W[j][k] > W[j][k_best]:
            if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

