import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from models.tools import *
from models.loss import AAMsoftmax
from models.models import ECAPA_TDNN
import numpy as np
import soundfile as sf
 
class ECAPAModel(nn.Module):
	def __init__(self, device='cuda:0', lr=0.001, lr_decay=0.97, C=512 , n_class=5994, m=0.2, s=30, test_step=1, **kwargs):
		super(ECAPAModel, self).__init__()
		self.device = device

		self.speaker_encoder = ECAPA_TDNN(C = C).to(self.device)

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
	def compute_ECAPAembedding(self, file_path):
		self.eval()
		torch.manual_seed(0)
		np.random.seed(0)
		audio, _ = sf.read(file_path)
		data = audio[np.newaxis, ...]
		data = torch.FloatTensor(data).to(self.device)
		embedding = self.speaker_encoder.forward(data, aug=False)

		return embedding.squeeze().cpu().numpy()
	
if __name__ == '__main__':
        model = ECAPAModel()
        model.load_parameters('exps/mytrain_ECAPA.model')