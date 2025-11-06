# coding:utf8
import warnings
import torch as t
from config.config import opt
class DefaultConfig(object):
    def __init__(self):
        self.env = 'default' 
        self.vis_port =8097 
        self.model_use = 'ECAPA_CNNModel'
        self.ref_choice = "GCC_PATH"
        self.n_speaker = 2
        self.cluster_method = "FCM"

        self.train_data_root = './data/train/' 
        self.test_data_root = './data/test1' 
        self.load_model_path = None

        self.batch_size = 32  # batch size
        self.use_gpu = True  # user GPU or not
        self.num_workers = 4  # how many workers for loading data
        self.print_freq = 20  # print info every N batch

        self.debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
        self.result_file = 'result.csv'

        self.max_epoch = 20
        self.lr = 0.001  # initial learning rate
        self.lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
        self.weight_decay = 0e-5  
        self.C = 512
        self.device = opt.device

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        print('user config:')
        for k, v in vars(self).items():
            if not k.startswith('_'):
                print(k, v)

ECAPA_opt = DefaultConfig()
