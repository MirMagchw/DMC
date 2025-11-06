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
        self.batch_size = 64      # Batch size
        self.use_gpu = True       # user GPU or not
        self.n_cpu = 8           # Number of loader threads
        self.print_freq = 20  # print info every N batch
        self.test_step = 1       # Test and save every [test_step] epochs
        self.lr = 0.001          # Learning rate
        self.lr_decay = 0.97    
        self.train = True

        ## Training and evaluation paths
        self.train_list = "../vox/train_list.txt"
        self.train_path = "../vox/voxceleb2/"
        self.eval_list = "../vox/vox1_list.txt"
        self.eval_path = "../vox/voxceleb1/"
        self.musan_path = "../vox/musan"
        self.rir_path = "../vox/RIRS_NOISES/simulated_rirs"
        self.save_path = "exps/exp1"
        self.model_save_path = os.path.join(self.save_path, 'model')
        self.score_save_path = os.path.join(self.save_path, 'score.txt')
        self.initial_model = ''

        ## Model and Loss parameters
        self.num_class = 5
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
                
CRNN_opt = DefaultConfig()
