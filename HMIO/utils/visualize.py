# import visdom
import time
import numpy as np
from tensorboard_logger import Logger

class Visualizer():
    '''
    封装了visdom，tensorboard_logger, 更方便记录loss
    '''
    def __init__(self, env='default',log_dir='runs/BiGRU', **kwargs):
        # self.vis = visdom.Visdom(env=env, **kwargs)
        self.tenbd = Logger(log_dir, flush_secs=2)

        # 记录数据的横向坐标{'img':2, 'loss':12}
        self.index = {}
        # 记录一些log信息
        self.log_text = ''
        
    # def reinit(self, env='default', **kwargs):
    #     '''
    #     更改visdom的配置
    #     '''
    #     self.vis =  visdom.Visdom(env=env, **kwargs)

        # return vis

    def plot(self, name, y):
        '''
        self.plot('loss',0.23)
        '''
        x = self.index.get(name, 0)
        # self.vis.line(Y=np.array([y]),
        #         X=np.array([x]),
        #         win=name,
        #         opts=dict(title=name),
        #         update=None if x==0 else 'append')
        self.tenbd.log_value(name,y,x)
        
        self.index[name] = x + 1

    def plotMany(self, data):
        '''
        一次渲染多个数据
        '''
        for k, v in data.iteritems():
            self.plot(k,v)
    
    def  log(self,info,win='log_text'):
        '''
        self.log({'loss':1, 'lr':0.001})
        '''
        # self.log_text += (
        #         '[{time}] {info} <br>'.format(
        #             time=time.strftime('%m%d_%H%M%S'),
        #             info=info
        #             )
        #         )
        # self.vis.text(self.log_text, win='log_text')

    # def __getattr__(self,name):
    #     return getattr(self.vis, name)
