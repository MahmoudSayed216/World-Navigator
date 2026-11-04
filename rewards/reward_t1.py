import torch
from utils import utils



class BasicReward:
    def __init__(self, theta, beta, C, K):
        self.theta = theta   # progress weight
        self.beta = beta     # per-step cost
        self.C = C           # goal reward
        self.K = K           # collision penalty

    # past_agent, curr_agent: (B, 3) = [x, y, uom], normalized
    # goal_xy:                (B, 2) absolute goal position, same normalization
    # termination_state:      (B, 1) processed values {0, 0.5, 1.0}
    def reward(self, past_agent, curr_agent, goal_xy, termination_state):
        term = (termination_state.view(-1) * 2).round()    # {0, 1, 2}, shape (B,)

        d_past = torch.linalg.norm(goal_xy - past_agent[:, :2], dim=-1)
        d_curr = torch.linalg.norm(goal_xy - curr_agent[:, :2], dim=-1)
        progress = d_past - d_curr

        r = self.calc_reward(term == 2, term == 1, progress)

        unchanged = torch.all(past_agent[:, :2] == curr_agent[:, :2], dim=1)   # (B,)
        return torch.where(unchanged, torch.zeros_like(r), r)

    # returns one reward per world, shape (B,)
    def calc_reward(self, reached_goal, collided, progress):
        return (self.C * reached_goal.float()
                - self.K * collided.float()
                + self.theta * progress
                - self.beta)