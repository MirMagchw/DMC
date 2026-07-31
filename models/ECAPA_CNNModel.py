import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from models.tools import *
from models.loss import AAMsoftmax
from models.models import ECAPA_CNN_TDNN
from models.batch_utils import prepare_waveform_batch
import numpy as np
import soundfile as sf

class ECAPA_CNNModel(nn.Module):
	def __init__(self, device, lr=0.001, lr_decay=0.97, C=512 , n_class=5994, m=0.2, s=30, test_step=1, **kwargs):
		super(ECAPA_CNNModel, self).__init__()
		self.device = device

		self.speaker_encoder = ECAPA_CNN_TDNN(C = C).to(self.device)

		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).to(self.device)

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
	def compute_ECAPAembedding_batch(self, file_paths):
		self.eval()
		if len(file_paths) == 0:
			return np.empty((0, 192), dtype=np.float32)

		data = prepare_waveform_batch(file_paths, self.device)
		embedding = self.speaker_encoder.forward(data, aug=False)
		return embedding.detach().cpu().numpy()
	@torch.no_grad()
	def compute_ECAPAembedding(self, file_path):
		return self.compute_ECAPAembedding_batch([file_path])[0]
