import torch
from utils import utils



class BasicReward:
    def __init__(self, theta, beta, C, K):
        self.theta = theta
        self.beta = beta
        self.C = C
        self.K = K
                    # B, 3              B, 3                B, MAX, 6         B, 2          B
    def reward(self, agent_past_state, agent_current_state, termination_state):
        if agent_past_state == agent_current_state:
            return 0 
        d = torch.linalg.norm(agent_past_state[:, :2] - agent_current_state[:, :2], dim=-1)
        
        return self.calc_reward(termination_state == 1, termination_state == 2, d)
        
        
        
    #RETURNS A VECTOR OF REWARDS PER WORLD PER STEP of shape B
    def calc_reward(self, reached_goal, collided, delta_dist):
        return self.C * reached_goal - self.K*collided + self.theta*delta_dist - self.beta
    