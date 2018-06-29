# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: PyCharm Community Edition
@file: data_helper.py
@time: 17/05/03 12:01
"""

import os
import json
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from . import transforms


class UGCDataset(Dataset):
    def __init__(self, file_path, voc_path, pos_path, cv=None):
        
        # 筛选数据
        filter_if = False
        self.df = pd.read_pickle(file_path)

        if cv and isinstance(cv,int):
            filter_if = self.df['cv']==cv
        elif isinstance(cv, list):
            for num in cv:
                filter_if |= (self.df['cv']==num)
        self.df = self.df[filter_if].reset_index(drop=True)

        # [(word, pos)]
        self.contents = self.df['Cut']
        self.sentence_lens = self.df['Len']
        # auxiliary labels
        self.auxiliary_labels = [
                self.df['Event'], self.df['Agent'], self.df['Object']
                ]
        # final labels
        self.top_labels = [
                self.df['Satisfaction'], 
                self.df['Disappointment'],
                self.df['Admiration'],
                self.df['Reproach'],
                self.df['Like'],
                self.df['Dislike']
                ]

        # load directly if voc file exists.
        if os.path.isfile(voc_path) and os.path.isfile(pos_path):
            with open(voc_path, 'r') as fv, open(pos_path, 'r') as fp:
                voc_dict = json.load(fv)
                voc = voc_dict['voc']
                max_length = voc_dict['max_length']

                pos_dict = json.load(fp)
                pos = pos_dict['voc']
        else:
            voc, pos, max_length = build_vocab(file_path, voc_path, pos_path)
        
        # index -> pad -> toTensor
        self.word_transform = transforms.Compose([
            transforms.ToIndex(voc),
            transforms.Pad(max_length),
            torch.LongTensor
            ])

        self.pos_transform = transforms.Compose([
            transforms.ToIndex(pos),
            transforms.Pad(max_length),
            torch.LongTensor
            ])


    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        sample = MultiLabelSample(
                content = self.contents[idx],
                wds = [word for word, _ in self.contents[idx]],
                pos = [pos for _, pos in self.contents[idx]],
                sentence_len = self.sentence_lens[idx],
                top_label = torch.FloatTensor([int(cate[idx]) for cate in self.top_labels]),
                bottom_label = torch.FloatTensor([int(cate[idx]) for cate in self.auxiliary_labels]),
                cv_n = self.df['cv'][idx]
                )
        # transforms to pytorch tensor
        sample.word_vec = self.word_transform(sample.wds)
        sample.pos_vec = self.pos_transform(sample.pos)

        return sample.__dict__
        


class MultiLabelSample(object):
    """ 多层标记数据样本
    @self.content: list(tuple) - tuple(word, word_type)
    @self.sentence_len: int - Num of words of sentence
    @self.top_label: str - The top label of data
    @self.bottom_label: str - The bottom label of data
    @self.split: int - k folds set index
    @self.vec: list - feature vector
    标记对应映射关系
    top_label_map = ['event', 'agent', 'object']
    bottom_label_map = ['Satisfaction', 'Disappointment',
        'Admiration', 'Reproach', 'Like', 'Dislike']
    """

    def __init__(self, 
            content, 
            wds, 
            pos, 
            sentence_len, 
            top_label, 
            bottom_label, 
            cv_n):

        self.top_label_map = ['Event', 'Agent', 'Object']
        self.bottom_label_map = [
            'Satisfaction', 'Disappointment', 'Admiration', 'Reproach', 'Like', 'Dislike'
        ]
        
        self.content = content
        self.wds = wds
        self.pos = pos
        self.sentence_len = sentence_len
        self.top_label = top_label
        self.bottom_label = bottom_label
        self.cv_n = cv_n
        self.top_probs = ''
        self.bottom_probs = ''

        # placeholder for vec
        self.word_vec = None
        self.pos_vec = None

    def __str__(self):
        return 'top_label_map:\t{}\ntop_label:\t{}\nbottom_label_map:\t{}\nbottom_label:\t\t{}\nsentence_len:\t{}\ncv_n:\t{}\ncontent:\t{}\nwds:\t{}\n'.format(
            '\t'.join(self.top_label_map), '\t'.join(map(str, self.top_label)),
            '\t'.join(self.bottom_label_map),
            '\t\t'.join(map(str, self.bottom_label)), self.sentence_len,
            self.cv_n, self.content, self.wds)

    def __len__(self):
        return self.sentence_len


def build_vocab(file_path, voc_path, pos_path):
    '''
    建立词典
    :param file_path: 原始文件地址
    :param voc_path: 词典路径
    :param pos_path: 词性路径
    :return:
    '''
    print('build vocab...')
    voc = set([])
    pos = set([])
    max_length = 0

    pd_data = pd.read_pickle(file_path)
    for i in range(pd_data.shape[0]):
        content = pd_data['Cut'][i]
        wds = [word for word, _ in content]
        poss = [p for _, p in content]

        max_length = max(max_length, len(wds))

        voc |= set(wds)
        pos |= set(poss)

    # set index
    voc = {wd: v for v, wd in enumerate(['<s>'] + list(voc))}
    pos = {p: j for j, p in enumerate(['<s>'] + list(pos))}
    print('build vocab done')

    if max_length > 100:
        max_length = 100

    voc_dict = {
        'voc': voc,
        'max_length': max_length,
    }
    pos_dict = {
        'voc': pos,
        'max_length': max_length,
    }
    with open(voc_path, 'w') as f:
        json.dump(voc_dict, f)

    with open(pos_path, 'w') as f:
        json.dump(pos_dict, f)

    return [voc, pos, max_length]


def build_data_cv(file_path, voc_path, pos_path, cv=5):
    '''
    从文件载入数据， 每个样本是一个HMultLabelSample对象
    :param file_path:  原始路径
    :param voc_path:   词典路径
    :param pos_path:   词性路径
    :param cv:         几折交叉验证
    :return:
    '''
    pd_data = pd.read_pickle(file_path)
    rev = []
    if os.path.isfile(voc_path) and os.path.isfile(pos_path):
        with open(voc_path, 'r') as fv, open(pos_path, 'r') as fp:
            voc_dict = json.load(fv)
            voc = voc_dict['voc']
            max_length = voc_dict['max_length']

            pos_dict = json.load(fp)
            pos = pos_dict['voc']
    else:
        voc, pos, max_length = build_vocab(
            file_path, voc_path, pos_path)

    print('length of voc: ', len(voc))
    print('length of pos: ', len(pos))

    bad_sampl_n = 0
    for i in range(pd_data.shape[0]):
        if i % 1000 == 0:
            print('load data:...', i)

        content = pd_data['Cut'][i]

        wds = [word for word, _ in content]
        poss = [pos for _, pos in content]

        # padding wds and pos
        pad_wds, pad_pos = wds[:max_length], poss[:max_length]
        pad_wds = [voc[x] if x in voc else 0 for x in pad_wds]
        pad_wds.extend([0] * (max_length - len(pad_wds)))
        pad_pos = [pos[x] if x in pos else 0 for x in pad_pos]
        pad_pos.extend([0] * (max_length - len(pad_pos)))

        # 句子长度，包括标点
        sentence_len = pd_data['Len'][i]

        # 词个数
        wds_cnt = len(wds)

        top_label = [
            pd_data['Event'][i], pd_data['Agent'][i], pd_data['Object'][i]
        ]

        bottom_label = [
            pd_data['Satisfaction'][i], pd_data['Disappointment'][i],
            pd_data['Admiration'][i], pd_data['Reproach'][i],
            pd_data['Like'][i], pd_data['Dislike'][i]
        ]
        if sum(bottom_label) == 0 or sum(top_label) == 0:
            bad_sampl_n += 1
            continue
        cv_n = np.random.randint(0, cv)
        datum = MultiLabelSample(content=content, raw_sentence=''.join(wds), wds=pad_wds, pos=pad_pos, wds_cnt=wds_cnt, sentence_len=sentence_len, top_label=top_label, bottom_label=bottom_label, cv_n=cv_n)
        rev.append(datum)
    print('bad sample number； {}'.format(bad_sampl_n))
    return rev, voc, pos, max_length


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z(),!?！？，。；’‘“”’\'\`]", " ", string)

    return string.strip().lower()

if __name__ == "__main__":
    ugc = UGCDataset(file_path='../../docs/data/HML_data_clean.dat', voc_path='../../docs/data/voc.json', pos_path='../../docs/data/pos.json')
    dataLoader = DataLoader(ugc, batch_size=4, shuffle=True)
    for batch in dataLoader:
        print(1)
        exit()
