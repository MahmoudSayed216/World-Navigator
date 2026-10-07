import torch.nn as nn
import torch


class RewardFunction(nn.Module):
	def __init__(self, alpha, beta, gamma, safety_threshold=0.15, critical_threshold = 0.05, grid_wh = (600, 600), eps = 1e-8):
		super(RewardFunction, self).__init__()
		assert abs(beta) <= gamma
		assert abs(alpha) <= gamma
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma 
		self.safety_threshold = safety_threshold
		self.critical_threshold = critical_threshold
		self.eps = eps
		self.width, self.height = grid_wh
		self.magnitude = (self.width**2 + self.height**2)**0.5

	def compute_wall_punishment(self, dw, dimension_length):
		
		dw /= dimension_length
		punishment = 0

		#* I believe alpha was a scaler greater than 1
		if dw < self.critical_threshold:
			punishment = -2*self.alpha * (self.safety_threshold - dw)
		elif dw < self.safety_threshold:
			punishment = -self.alpha * (self.safety_threshold - dw)

		return punishment


	def compute_obstacle_punishment(self, do):
		pass
		
	
	def forward(self, dws, dos, df):
		"""
		dws will come in [L, R, U, D] order
		"""
		wall_reward = 0
		wall_reward += self.compute_wall_punishment(dws[0], self.width)
		wall_reward += self.compute_wall_punishment(dws[1], self.width)
		wall_reward += self.compute_wall_punishment(dws[2], self.height)
		wall_reward += self.compute_wall_punishment(dws[3], self.height)


		obstacle_reward = 0
		obstacle_reward += self.compute_obstacle_punishment(dos[0])
		obstacle_reward += self.compute_obstacle_punishment(dos[1])
		obstacle_reward += self.compute_obstacle_punishment(dos[2])
		obstacle_reward += self.compute_obstacle_punishment(dos[3])


