# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:14:46 2016

@author: Balázs Hidasi
"""

import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
import gru4rec
import evaluation
import pickle 
import os

PATH_TO_TRAIN = '/content/GRU4Rec/DynamicRec/nowplaying_grurec_train_data.txt'
PATH_TO_TEST = '/content/GRU4Rec/DynamicRec/nowplaying_grurec_test_data.txt'

if __name__ == '__main__':
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId':np.int64})
    
    #State-of-the-art results on RSC15 from "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations" on RSC15 (http://arxiv.org/abs/1706.03847)
    #BPR-max, no embedding (R@20 = 0.7197, M@20 = 0.3157)
    gru = gru4rec.GRU4Rec(loss='bpr-max', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad', n_epochs=1, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2, momentum=0.3, n_sample=2048, sample_alpha=0, bpreg=1, constrained_embedding=False)
    if os.path.exists('saved_model.pkl'):
        print('loading')
        with open('saved_model.pkl', 'rb') as f:
          gru = pickle.load(f)
    else:
        print('fitting and saving')
        gru.fit(data)
        with open('saved_model.pkl', 'wb') as f:
          pickle.dump(gru, f)
    res = evaluation.evaluate_gpu(gru, valid, 20)
    print('Recall@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))

    res = evaluation.evaluate_gpu(gru, valid, 10)
    print('Recall@10: {}'.format(res[0]))
    print('MRR@10: {}'.format(res[1]))

    #BPR-max, constrained embedding (R@20 = 0.7261, M@20 = 0.3124)
    # gru = gru4rec.GRU4Rec(loss='bpr-max', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad', n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2, momentum=0.1, n_sample=2048, sample_alpha=0, bpreg=0.5, constrained_embedding=True)
    # gru.fit(data)
    # res = evaluation.evaluate_gpu(gru, valid)
    # print('Recall@20: {}'.format(res[0]))
    # print('MRR@20: {}'.format(res[1]))

    #Cross-entropy (R@20 = 0.7180, M@20 = 0.3087)
    # gru = gru4rec.GRU4Rec(loss='cross-entropy', final_act='softmax', hidden_act='tanh', layers=[100], adapt='adagrad', n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0.3, learning_rate=0.1, momentum=0.7, n_sample=2048, sample_alpha=0, bpreg=0, constrained_embedding=False)
    # gru.fit(data)
    # res = evaluation.evaluate_gpu(gru, valid)
    # print('Recall@20: {}'.format(res[0]))
    # print('MRR@20: {}'.format(res[1]))
    
    #OUTDATED!!!
    #Reproducing results from the original paperr"Session-based Recommendations with Recurrent Neural Networks" on RSC15 (http://arxiv.org/abs/1511.06939)
    #print('Training GRU4Rec with 100 hidden units')    
    #gru = gru4rec.GRU4Rec(loss='top1', final_act='tanh', hidden_act='tanh', layers=[100], batch_size=50, dropout_p_hidden=0.5, learning_rate=0.01, momentum=0.0, time_sort=False)
    #gru.fit(data)
    #res = evaluation.evaluate_gpu(gru, valid)
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))
