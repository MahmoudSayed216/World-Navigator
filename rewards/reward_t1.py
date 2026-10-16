import torch



def compute_reward(reached_goal, collided, theta, beta, prev_dist, curr_dist):
    C = 10
    K = 10

    