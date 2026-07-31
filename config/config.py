# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    def __init__(self):
        self.env = 'default'  
        self.vis_port =8097 
        self.model_use = 'CRNN_featureModel(60)'  
        self.ref_choice = 'GCC_PHAT'
        self.n_speaker = 3 
        self.snr = 10
        self.t60 = 400
        self.cluster_method = 'FCM'

        self.debug_file = '/tmp/debug'  
        self.result_file = 'result.csv'

        self.device = t.device('cuda:0') if t.cuda.is_available() and self.use_gpu else t.device('cpu')
    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        self.device = t.device('cuda:0') if t.cuda.is_available() and self.use_gpu else t.device('cpu')
        print('user config:')
        for k, v in vars(self).items():
            if not k.startswith('_'):
                print(k, v)

opt = DefaultConfig()
