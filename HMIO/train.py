# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: PyCharm Community Edition
@file: adios_train.py
@time: 17/05/03 17:39
"""

import json
import os
import time
from math import ceil

import numpy as np
from sklearn import linear_model as lm
from utils.metrics import (Average_precision, Coverage, Hamming_loss,
                           One_error, Ranking_loss, Construct_thresholds,
                           F1_measure)

 
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from config import params
from utils import hiso
from utils.data_helper import *
from utils.visualize import Visualizer

# 设置仅使用第二块gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

vis = Visualizer(env='default',port=8099,log_dir="runs/%s"%time.strftime("%m-%d-%H:%M:%S", time.localtime()))
use_cuda = torch.cuda.is_available()



def train(dataloader,testloader):
    '''
    训练模型入口
    '''
    # build model
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    # 保存最优模型
    # model_dir = params.model_dir + time.strftime("%Y-%m-%d-%H:%M:%S",
    #                                                time.localtime())
    # s.mkdir(model_dir)
    # model_name = model_dir + '/' + params.model_name
    # build model
    model = hiso.HISO(params)
    margin_loss = hiso.HisoLoss(params)
    if use_cuda: 
        model.cuda()
        margin_loss.cuda()
    # learning rate
    lr = params.lr
    optimizer = optim.RMSprop(model.parameters(), lr=lr)#,momentum=0.9,weight_decay=0.01)
    scheduler = MultiStepLR(optimizer, milestones=[int(0.3*params.epochs),int(0.7*params.epochs)], gamma=0.1)
    # criterion = torch.nn.BCELoss()
    model.train()

    a_probs,f_probs,a_labels,f_labels= None, None,None,None
    total_loss = []
    for epoch in range(params.epochs):
        # set lr dynamicly
        scheduler.step()
        for batch_idx, samples in enumerate(dataloader, 0):
            v_word = Variable(samples['word_vec'].cuda() if use_cuda else samples['word_vec'])
            v_pos = Variable(samples['pos_vec'].cuda() if use_cuda else samples['pos_vec'])

            v_auix_label = Variable(samples['bottom_label'].cuda() if use_cuda else samples['bottom_label'])
            v_final_label = Variable(samples['top_label'].cuda() if use_cuda else samples['top_label'])

            final_probs, auxi_probs = model(v_word, v_pos)
            # autograd optim
            optimizer.zero_grad()
            loss = margin_loss(auxi_probs, v_auix_label, final_probs, v_final_label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(),0.9)
            optimizer.step()
            
            total_loss.append(loss.data[0])
            
            # log 平滑记录
            if a_probs is None:
                a_probs,a_labels = auxi_probs.data.cpu().numpy(),v_auix_label.data.cpu().numpy()
                f_probs,f_labels = final_probs.data.cpu().numpy(),v_final_label.data.cpu().numpy()
            else:
                a_probs,a_labels = np.vstack([a_probs,auxi_probs.data.cpu().numpy()]),\
                        np.vstack([a_labels,v_auix_label.data.cpu().numpy()])
                f_probs,f_labels = np.vstack([f_probs,final_probs.data.cpu().numpy()]),\
                        np.vstack([f_labels,v_final_label.data.cpu().numpy()])

            # evaluate train model
            if batch_idx % params.log_interval == 1:
                vis.plot('margin loss',np.mean(total_loss))
                vis.log('margin loss: %s'%np.mean(total_loss), win='marginLoss_text')
                vis_log(f_probs, f_labels, name='train final',numpy_data=True)
                vis_log(a_probs, a_labels, name='train auxiliary',numpy_data=True)
                # reinit
                total_loss = []
                a_probs,f_probs,a_labels,f_labels= None, None,None,None


            if batch_idx % (params.log_interval * 5) == 1:
                evaluate(model,testloader,margin_loss)
                # chanage model
                model.train()
                

def evaluate(model, dataloader, margin_loss):

    model.eval()
    a_probs,f_probs,a_labels,f_labels= None, None,None,None
    loss = 0.
    for batch_idx, samples in enumerate(dataloader, 0):
        v_word = Variable(samples['word_vec'].cuda() if use_cuda else samples['word_vec'])
        v_pos = Variable(samples['pos_vec'].cuda() if use_cuda else samples['pos_vec'])

        v_auix_label = Variable(samples['bottom_label'].cuda() if use_cuda else samples['bottom_label'])
        v_final_label = Variable(samples['top_label'].cuda() if use_cuda else samples['top_label'])

        final_probs, auxi_probs = model(v_word, v_pos)
        loss += margin_loss(auxi_probs,v_auix_label,final_probs,v_final_label).data[0]

        if batch_idx == 0:
            a_probs,a_labels = auxi_probs.data.cpu().numpy(),v_auix_label.data.cpu().numpy()
            f_probs,f_labels = final_probs.data.cpu().numpy(),v_final_label.data.cpu().numpy()
        else:
            a_probs,a_labels = np.vstack([a_probs,auxi_probs.data.cpu().numpy()]),\
                    np.vstack([a_labels,v_auix_label.data.cpu().numpy()])
            f_probs,f_labels = np.vstack([f_probs,final_probs.data.cpu().numpy()]),\
                    np.vstack([f_labels,v_final_label.data.cpu().numpy()])
    # log
    vis.plot('test margin loss',loss/batch_idx)
    vis_log(f_probs, f_labels, name='test final',numpy_data=True)
    vis_log(a_probs, a_labels, name='test auxiliary',numpy_data=True)

    

def vis_log(probs, labels, name='', numpy_data=False):
    '''
    记录多标签评价指标信息
    '''
    if not numpy_data:
        labels = labels.data.cpu().numpy()
        probs = probs.data.cpu().numpy()

    preds = (probs>0.5).astype(np.float32)
    # print(labels[0],'\n',probs[0],'\n',preds[0])

    rk_loss = Ranking_loss(labels, probs)
    one_error = One_error(labels, probs)
    hm_loss = Hamming_loss(labels,preds)
    f1_micro = F1_measure(labels,preds,average='micro')
    f1_macro = F1_measure(labels,preds,average='macro')
    cover = Coverage(labels,probs)
    ap = Average_precision(labels,probs)
    # visdom 
    vis.plot('%s Ranking Loss'%name, rk_loss)
    vis.plot('%s One Error'%name, one_error)
    vis.plot('%s Hamming Loss'%name, hm_loss)
    vis.plot('%s F1@micro'%name, f1_micro)
    vis.plot('%s F1@macro'%name, f1_macro)
    vis.plot('%s Coverage'%name, cover)
    vis.plot('%s Average Precision'%name, ap)
            

if __name__ == '__main__':
    # load params
    trainset = UGCDataset(file_path='../docs/data/HML_data_clean.dat',
            voc_path='../docs/data/voc.json',
            pos_path='../docs/data/pos.json',
            cv=list(range(8)))
    
    train_loader = DataLoader(trainset,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.num_workers)
    
    testset = UGCDataset(file_path='../docs/data/HML_data_clean.dat',
            voc_path='../docs/data/voc.json',
            pos_path='../docs/data/pos.json',
            cv=[8,9])
    
    test_loader = DataLoader(testset,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.num_workers)
    train(train_loader, test_loader)
