from model.model import RLModel
from model.reward_function import RewardFunction
from environment import Environment
from agent import Agent

MAX_STEPS = 1000
EPISODES = 1000
N_ENVS = 30
gamma = 1
GAMMA_SCALING_FACTOR = 0.99
T_value_per_episode = [[1 for _ in range(round(0.08*MAX_STEPS))] + # constant for initial learning
                        [1 for _ in range(round(0.3*MAX_STEPS))] + # decreasing
                        [1 for _ in range(round(0.55*MAX_STEPS))] + 
                        [1 for _ in range(round(0.07*MAX_STEPS))]
                          ]

for episode in range(EPISODES):
    done = False
    n_steps = 0
    # define N different environments each with a different initial state
    while not done and n_steps <= MAX_STEPS:
        # pass observations of N envs to model
        # get model's output for N envs
        # for env in envs:
            # if ith env done:
                # reward = big_number, stop passing this env to the model
            # if ith env is not done :
                # update state of ith env
                # calculate reward based on current state
                # accumulate reward  [Don't update the model yet]
    
    
    # update model based on cumulative reward after the episode is over 

        n_steps+=1