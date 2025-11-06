# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    def __init__(self):
        self.env = 'default'  
        self.vis_port =8097 
        self.model_use = 'CRNN_featureModel(60)'  
        self.ref_choice = 'FCM'
        self.n_speaker = 2 
        self.snr = 5
        self.t60 = 666
        self.cluster_method = 'CountNet'

        self.train_data_root = './data/train/' 
        self.test_data_root = './data/test1' 
        self.load_model_path = None  

        self.batch_size = 32  
        self.use_gpu = True  
        self.num_workers = 4  
        self.print_freq = 20  

        self.debug_file = '/tmp/debug'  
        self.result_file = 'result.csv'

        self.max_epoch = 20
        self.lr = 0.001  
        self.lr_decay = 0.5  
        self.weight_decay = 0e-5  
        self.C = 512
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
