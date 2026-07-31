# coding:utf8
import warnings, os
import torch as t
from config.config import opt
class DefaultConfig(object):
    def __init__(self):
        self.env = 'default' 
        self.vis_port = 8097 
        self.model = 'CRNN' 
        self.wav_path = ''
        ## Training parameters
        self.num_frames = 400     # Duration of the input segments
        self.max_epoch = 100      # Maximum number of epochs
        self.batch_size = 64   # Batch size
        self.use_gpu = True       # user GPU or not
        self.n_cpu = 8           # Number of loader threads
        self.print_freq = 20  # print info every N batch
        self.test_step = 1       # Test and save every [test_step] epochs
        self.lr = 0.001          # Learning rate
        self.lr_decay = 0.97     # Learning rate decay every [test_step] epochs
        self.train = True

        ## Training and evaluation paths
        self.train_list = "..."
        self.train_path = "..."
        self.eval_list = "..."
        self.eval_path = "..."
        self.musan_path = "..."
        self.rir_path = "..."
        self.save_path = "..."
        self.model_save_path = os.path.join(self.save_path, 'model')
        self.score_save_path = os.path.join(self.save_path, 'score.txt')
        self.initial_model = ''

        ## Model and Loss parameters
        self.num_class = 5994
        self.input_channels = 1
        self.device = opt.device
    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        os.makedirs(self.model_save_path, exist_ok=True)

        print('user config:')
        for k, v in vars(self).items():
            if not k.startswith('_'):
                print(k, v)
                
CRNN_feature_opt = DefaultConfig()
