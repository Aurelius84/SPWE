#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.squish_w = nn.Parameter(torch.randn(2 * opt.ghid_size, 2 * opt.ghid_size))
        self.atten_proj = nn.Parameter(torch.randn(2 * opt.ghid_size, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, weight, mask=None, **kwargs):
        # print(input)
        _squish = batch_matmul(weight, self.squish_w, active_func='tanh')
        # print("_squish", _squish)
        att_weight = batch_matmul(_squish, self.atten_proj)
        # print("att_weight", att_weight)

        if mask is not None:
            att_weight_norm = softmax_mask(att_weight, mask, axis=-1)
        else:
            att_weight_norm = self.softmax(att_weight)
        # print('+++')
        # print(input, att_weight_norm)
        _out = attention_matmul(input, att_weight_norm)

        return _out


def batch_matmul(seq, weight, active_func=''):
    '''
    seq matmul weight with keeping dims
    '''
    out = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if active_func == 'tanh':
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)

        out = _s if out is None else torch.cat((out, _s), 0)
    return out


def attention_matmul(seq, att_weight):
    '''
    apply att_weight on seq
    '''
    att_out = []
    for i in range(seq.size(0)):
        s_i = att_weight[i] * seq[i].transpose(1,0)
        att_out.append(torch.sum(s_i, 1).view(1,-1))

    return torch.cat(att_out, dim=0)


def softmax_mask(input, mask, axis=1, epsilon=1e-12):
    input = input.squeeze()
    assert input.size() == mask.size()

    shift, _ = torch.max(input, axis, keepdim=True)
    shift = cudaWrapper(shift.expand_as(input))

    target_exp = torch.exp(input - shift) * Variable(mask)

    normalize = torch.sum(target_exp, axis, keepdim=True).expand_as(target_exp)
    softm = target_exp / (normalize + epsilon)

    return cudaWrapper(softm)

def cudaWrapper(input):
    # return input
    return input.cuda() if torch.cuda.is_available() else input