import torch, sys, os, tqdm, numpy, soundfile, time, pickle, librosa
import pandas as pd
import torch.nn as nn
from torchnet import meter
from models.models import CRNN
import numpy as np
import soundfile as sf

eps = np.finfo(float).eps
# meters
loss_meter = meter.AverageValueMeter()
confusion_matrix = meter.ConfusionMeter(5)
previous_loss = 1e10

class CRNNModel(nn.Module):
    def __init__(self, lr, num_class, input_channels, test_step, lr_decay, device, print_freq=20, **kwargs):
        super(CRNNModel, self).__init__()
        self.print_freq = print_freq
        self.device = device
        self.crnn = CRNN(num_classes=num_class, input_channels=input_channels).to(self.device)
        self.activation = nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
        self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        map_location = self.device
        loaded_state = torch.load(path, map_location=map_location)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

    @torch.no_grad()
    def count(self, file_path):
        self.eval()
    
        torch.manual_seed(0)
        np.random.seed(0)
        audio, _ = sf.read(file_path)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        X = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T
        if X.shape[0] < 400:
            pad_width = ((0, 400 - X.shape[0]), (0, 0))
            X = np.pad(X, pad_width, mode='constant', constant_values=0)
        elif X.shape[0] > 400:
            X = X[:400, :]
        Theta = np.linalg.norm(X, axis=1) + eps
        X /= np.mean(Theta)
        X = X[np.newaxis, np.newaxis, ...]
        processed_batch = torch.FloatTensor(X).to(self.device)
        score = self.crnn(processed_batch)
        score = self.activation(score).squeeze().cpu().numpy()
        return score