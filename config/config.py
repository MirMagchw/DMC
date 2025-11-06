# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    def __init__(self):
        self.env = 'default'  # visdom 环境
        self.vis_port =8097 # visdom 端口
        self.model_use = 'CRNN_featureModel(60)'  # 使用的模型，名字必须与models/__init__.py中的名字一致
        self.ref_choice = 'FCM'
        self.n_speaker = 2  # 说话人数量
        self.snr = 5
        self.t60 = 666
        self.cluster_method = 'CountNet'

        self.train_data_root = './data/train/'  # 训练集存放路径
        self.test_data_root = './data/test1'  # 测试集存放路径
        self.load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

        self.batch_size = 32  # batch size
        self.use_gpu = True  # user GPU or not
        self.num_workers = 4  # how many workers for loading data
        self.print_freq = 20  # print info every N batch

        self.debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
        self.result_file = 'result.csv'

        self.max_epoch = 20
        self.lr = 0.001  # initial learning rate
        self.lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
        self.weight_decay = 0e-5  # 损失函数
        self.C = 512
        self.device = t.device('cuda:0') if t.cuda.is_available() and self.use_gpu else t.device('cpu')
    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
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
