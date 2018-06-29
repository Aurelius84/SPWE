#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pack and unpack with mask function from: https://github.com/kevinkwl/AoAReader/blob/master/aoareader/AoAReader.py

# @Time    : 2017/12/6 11:13
# @From    : PyCharm
# @File    : hiso
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from utils.attention import Attention, cudaWrapper


def sort_batch(data, seq_len):
    sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True)
    sorted_data = data[sorted_idx.data]
    _, reverse_idx = torch.sort(sorted_idx, dim=0, descending=False)

    return sorted_data, cudaWrapper(sorted_seq_len), cudaWrapper(reverse_idx)


def create_mask(seq_lens):
    mask = torch.zeros(seq_lens.data.size(0), torch.max(seq_lens.data))
    for i, seq_len in enumerate(seq_lens.data):
        mask[i][:seq_len] = 1

    return cudaWrapper(mask.float())


def getSeqLength(input, return_variable=True):
    seq_len = torch.LongTensor([torch.nonzero(input.data[i]).size(0) for i in range(input.data.size(0))])
    if return_variable:
        seq_len = Variable(seq_len)
    return cudaWrapper(seq_len)


class HISO(nn.Module):
    def __init__(self, opt):
        super(HISO, self).__init__()

        self.model_name = 'HISO'
        self.opt = opt
        # Embedding Layer
        self.wd_embed = nn.Embedding(opt.voc_size, opt.embed_dim)
        self.pos_embed = nn.Embedding(opt.pos_size, opt.embed_dim)
        self.initEmbedWeight()
        # Bi-GRU Layer
        self.wd_bi_gru = nn.GRU(input_size = opt.embed_dim,
                hidden_size = opt.ghid_size,
                num_layers = opt.glayer,
                bias = True,
                batch_first = True,
                dropout = 0.5,
                bidirectional = True
                )
        self.attention = Attention(opt)

        # Bi-GRU Layer
        self.pos_bi_gru = nn.GRU(input_size = opt.embed_dim,
                hidden_size = opt.ghid_size,
                num_layers = opt.glayer,
                bias = True, 
                batch_first = True,
                dropout = 0.5,
                bidirectional = True
                )

        # output from pos hidden layer to predict middle labels
        pos_hidden_size = opt.ghid_size * opt.glayer
        self.pos_fc = nn.Sequential(
                nn.BatchNorm1d(pos_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(pos_hidden_size, opt.auxiliary_labels),
                nn.Sigmoid()
                )
        # predict final labels
        combine_size = opt.ghid_size * opt.glayer + opt.auxiliary_labels
        self.fc = nn.Sequential(
                nn.Linear(combine_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace = True),
                nn.Linear(128, opt.label_dim),
                nn.Sigmoid()
                )
        self.softmax = nn.Softmax(dim=1)

    def initEmbedWeight(self):
        '''
        init embedding layer from random|word2vec|sswe
        '''
        if 'w2v' in self.opt.init_embed:
            weights = Word2Vec.load('../docs/data/w2v_word_100d_5win_5min')
            voc = json.load(open('../docs/data/voc.json','r'))['voc']
            print(weights[list(voc.keys())[3]])

            word_weight = np.zeros((len(voc),self.opt.embed_dim))
            for wd,idx in voc.items():
                vec = weights[wd] if wd in weights else np.random.randn(self.opt.embed_dim)
                word_weight[idx] = vec
            # print(word_weight[3])
            self.wd_embed.weight.data.copy_(torch.from_numpy(word_weight))

            weights = Word2Vec.load('../docs/data/w2v_pos_100d_5win_5min')
            pos = json.load(open('../docs/data/pos.json','r'))['voc']
            pos_weight = np.zeros((len(pos),self.opt.embed_dim))
            for ps,idx in pos.items():
                vec = weights[ps] if ps in weights else np.random.randn(self.opt.embed_dim)
                pos_weight[idx] = vec
            self.pos_embed.weight.data.copy_(torch.from_numpy(pos_weight))

        elif 'sswe' in self.opt.init_embed:
            word_weight = pickle.load(open('../docs/model/%s'% self.opt.embed_path,'rb'))
            self.wd_embed.weight.data.copy_(torch.from_numpy(word_weight))
        # random default


    def forward(self, wd, pos):
        wd_len, pos_len = getSeqLength(wd), getSeqLength(pos)
        wd_mask, pos_mask = create_mask(wd_len), create_mask(pos_len)

        s_wd, s_wd_len, reverse_wd_idx = sort_batch(wd, wd_len)
        s_pos, s_pos_len, reverse_pos_idx = sort_batch(pos, pos_len)

        wd_embedding = pack(self.wd_embed(s_wd), list(s_wd_len.data), batch_first=True)
        pos_embedding = pack(self.pos_embed(s_pos), list(s_pos_len.data), batch_first=True)

        # Bi-GRU
        wd_out, _ = self.wd_bi_gru(wd_embedding)
        pos_out, _ = self.pos_bi_gru(pos_embedding)

        wd_out, _ = unpack(wd_out, batch_first=True)
        pos_out,_ = unpack(pos_out, batch_first=True)

        wd_out = wd_out[reverse_wd_idx.data]
        pos_out = pos_out[reverse_pos_idx.data]

        # attention
        if 'word' in self.opt.attention:
            wd_atten = self.attention(wd_out, weight=wd_out, mask=wd_mask)
        elif 'pos' in self.opt.attention:
            wd_atten = self.attention(wd_out, weight=pos_out, mask=pos_mask)
        else:
            wd_atten = wd_out[:,-1,:]
        
        # pos_out to predict auxiliary label
        auxi_probs = self.pos_fc(pos_out[:, -1, :].contiguous())

        # combine wd_out with auxi_probs as feature
        combine_feature = torch.cat((wd_atten, auxi_probs), dim=1)
        logits = self.fc(combine_feature)

        return logits, auxi_probs


    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.opt.glayer, batch_size, self.opt.ghid_size)
        return Variable(h0)



class HisoLoss(nn.Module):
    def __init__(self, opt):
        super(HisoLoss, self).__init__()
        self.opt = opt
        # self.reconstruction_loss = 0 // todo

    def forward(self,auxi_probs, auxi_labels, final_probs, final_labels):
        # calcu auxi_labels margin loss
        self.auxi_loss = self.marginLoss(auxi_probs, auxi_labels)

        # calcu final_labels margin loss
        self.final_loss = self.marginLoss(final_probs, final_labels)
        
        self.loss = self.opt.loss_alpha * self.auxi_loss + self.final_loss
        return self.loss

    def marginLoss(self, probs, labels):
        
        left = F.relu(self.opt.max_margin - probs, inplace=True)**2
        right = F.relu(probs - self.opt.min_margin, inplace=True)**2

        margin_loss = labels * left + (1. - labels) * right
        return margin_loss.sum() / labels.size(0)


class opt(object):
    voc_size = 100
    pos_size = 57
    embed_dim = 20
    ghid_size = 3
    seq_len = 4
    glayer = 2
    auxiliary_labels = 3
    label_dim = 6
    max_margin = 0.9
    min_margin = 0.1
    embed_path='lookup_01-22-19:10'
    init_embed='randn'
    loss_alpha=1e-2
    attention='word'


if __name__ == '__main__':
    import torch.optim as optim

    wd = Variable(torch.LongTensor([[2,45,75,0], [5,54,76,23]]))
    pos = Variable(torch.LongTensor([[3,45,8,0], [13,56,7,43]]))
    labels = Variable(torch.FloatTensor([[1,0,0,1,0,0],[0,0,1,0,1,0]]))
    auxi = Variable(torch.FloatTensor([[1,0,0],[0,1,0]]))

    model = HISO(opt)
    Loss = HisoLoss(opt)
    op = optim.SGD(model.parameters(),lr=0.1)
    model.train()
    for i in range(100):
        final_probs,auxi_probs = model(wd, pos)
        loss = Loss(auxi_probs, auxi, final_probs, labels)
        op.zero_grad()
        loss.backward()
        op.step()
        print(loss.data[0])
