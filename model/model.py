import torch.nn as nn


class RLModel(nn.Module):
	def __init__(self, n_observations, n_actions):
		super(RLModel, self).__init__()
		
		self.T = 1.0
		
		self.layer1 = nn.Sequential(
			nn.Linear(n_observations, 50),
			nn.BatchNorm1d(50),
			nn.Tanh()
		)

		self.layer2 = nn.Sequential(
			nn.Linear(50, 60),
			nn.BatchNorm1d(60),
			nn.Tanh()
		)

		self.output = nn.Sequential(
			nn.Linear(60, n_actions),
		)

		self.smx = nn.Softmax()

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.output(x)
		x = x/self.T
		x = self.smx(x)
		
		return x
	

	def set_temparature(self, T):
		self.T = T
	