import torch, sys, os, tqdm, numpy, soundfile, time, pickle, librosa
import pandas as pd
import torch.nn as nn
from torchnet import meter
from models.models import CRNN
from models.batch_utils import prepare_crnn_chunk_batch
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
    def count_batch(self, file_paths):
        self.eval()
        if len(file_paths) == 0:
            return np.empty((0, 5), dtype=np.float32)

        chunk_batch, chunk_counts = prepare_crnn_chunk_batch(file_paths)
        if chunk_batch.numel() == 0:
            return np.empty((0, 5), dtype=np.float32)

        chunk_scores = self.crnn(chunk_batch.to(self.device))
        outputs = []
        start = 0
        for count in chunk_counts.tolist():
            end = start + count
            file_score = chunk_scores[start:end].mean(dim=0, keepdim=True)
            file_score = self.activation(file_score)
            outputs.append(file_score.squeeze(0).detach().cpu())
            start = end

        return torch.stack(outputs, dim=0).numpy()

    @torch.no_grad()
    def count(self, file_path):
        return self.count_batch([file_path])[0]
