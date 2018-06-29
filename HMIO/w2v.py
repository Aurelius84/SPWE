#coding:utf-8

import time
import os
import gensim
import pandas as pd


class Dataset():
    def __init__(self,file_path,mode='word'):
        self.mode = mode
        self.df = pd.read_pickle(file_path)

    def __iter__(self):
 
        mode = 0 if 'word' in self.mode else 1
        for idx in range(len(self.df)):
            yield [tp[mode] for tp in self.df['Cut'][idx]]


def train_w2v(file_path, mode, save_dir):
    '''
    预训练word2vec
    '''
    data = Dataset(file_path,mode)
    start_t = time.time()
    print('start training....')
    model = gensim.models.Word2Vec(data,
                    min_count=5,
                    size=100,
                    window=5,
                    workers=4
                )
    print('complete! cost %d s'%(time.time()-start_t))

    model.save('%sw2v_%s_100d_5win_5min'%(save_dir,mode))


if __name__ == '__main__':
    data = gensim.models.Word2Vec.load('../docs/data/w2v_word_50d_5win_5min')
    print(data[u'喜欢'])
    exit()
    train_w2v(file_path='../docs/data/HML_data_clean.dat',
            mode='pos',
            save_dir='../docs/data/')





                

