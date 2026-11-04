from world import World
import random
import torch
from utils.utils import *

def create_worlds(num_worlds, worlds_height, worlds_width, min_obstacles, max_obstacles, minimum_distance_between_agent_and_goal,
                  maximum_distance_between_agent_and_goal, knowledge_type, agent_side_length, obstacle_constraints_shorter_side,
                    obstacle_constraints_longer_side, min_unit_of_motion, max_unit_of_motion, passage_space_factor):
    
    worlds = []
    distances = power_dist(minimum_distance_between_agent_and_goal, maximum_distance_between_agent_and_goal, num_worlds, 1.25)
    # print("distances: ",distances)
    for i in range(num_worlds):
        n_obst = int(round(random.uniform(min_obstacles, max_obstacles)))
        # uom = int(round(random.uniform(min_unit_of_motion, max_unit_of_motion)))
        world = World(
            world_height=                               worlds_height,
            world_width=                                worlds_width,
            n_obstacles=                                n_obst,
            minimum_distance_between_agent_and_goal=    distances[i],
            knowledge_type=                             knowledge_type,
            agent_side_length=                          agent_side_length,
            obstacle_constraints_shorter_side=          obstacle_constraints_shorter_side,
            obstacle_constraints_longer_side=           obstacle_constraints_longer_side,
            unit_of_motion=                             1,
            passage_space_factor=                       passage_space_factor
        )

        worlds.append(world)

    return worlds



def move_worlds_to_next_state(worlds: list[World], actions, uoms):
    new_states = []
    for world, action, uom in zip(worlds, actions, uoms):
        state = world.next_state(World.ACTIONS_MAP[action.item()], uom.item())
        new_states.append(state)

    return new_states

def get_worlds_current_state(worlds: list[World]):
    current_states = []
    for world in worlds:
        current_states.append(world.get_current_state())

    return current_states

def pad_states(obstacle_vectors, max_obstacles):
    
    for vec in obstacle_vectors:
        diff = int(max_obstacles - len(vec)/6)
        vec.extend([0]*6*diff)



def process_states_vector(states, max_obstacles, world_width, world_height, max_uom):
    # state = agents XYWH, goal XYWH, N Obstacles XYWHAD
    # return -> agents XY goal XY
    n_worlds = len(states)
    # print("len(state): ", len(states))
    # print("len(state[0]): ", len(states[0]))
    # print("STATE: \n", states)
    
    # change this cw stupid shit
    cw_agents = [] # list of lists
    cw_goals = [] # list of lists
    cw_obstacles = [] # list of list of lists
    cw_termination_states = []

    for i in range(n_worlds):
        state = states[i] # states of world i

        agent = state[:2] #xy
        agent.extend([state[4]]) #uom
        goal = state[5:7] #xy
        obstacles = state[10:-1] #xywh, axis, direction
        termination_state = state[-1]
        
        cw_agents.append(agent)
        cw_goals.append(goal)
        cw_obstacles.append(obstacles)
        cw_termination_states.append(termination_state)

    pad_states(cw_obstacles, max_obstacles)
    cw_agents = torch.tensor(cw_agents, dtype=torch.float32)
    cw_goals = torch.tensor(cw_goals, dtype=torch.float32)
    cw_obstacles = torch.tensor(cw_obstacles, dtype=torch.float32).reshape(n_worlds, max_obstacles, 6)
    cw_termination_states = torch.tensor(cw_termination_states, dtype=torch.float32).reshape(n_worlds, 1)/2.0

    mask = (cw_obstacles.abs().sum(-1, keepdim=True) > 0).float()  # before the obstacle edits

    cw_agents[:, 0] = cw_agents[:, 0] / world_width 
    cw_agents[:, 1] = cw_agents[:, 1] / world_height
    cw_agents[:, 2] = cw_agents[:, 2] / max_uom

    cw_goals[:, 0] = cw_goals[:, 0] / world_width 
    cw_goals[:, 1] = cw_goals[:, 1] / world_height
    cw_goals[:, 0] = cw_goals[:, 0] - cw_agents[:, 0]
    cw_goals[:, 1] = cw_goals[:, 1] - cw_agents[:, 1]

    cw_obstacles[:, :, [0, 2]] = (cw_obstacles[:, :, [0, 2]]/world_width)
    cw_obstacles[:, :, 0] = cw_obstacles[:, :, 0] - cw_agents[:, 0:1]
    cw_obstacles[:, :, [1, 3]] = (cw_obstacles[:, :, [1, 3]]/world_height)
    cw_obstacles[:, :, 1] = cw_obstacles[:, :, 1] - cw_agents[:, 1:2]
    cw_obstacles = cw_obstacles*mask

    return cw_agents, cw_goals, cw_obstacles, cw_termination_states.squeeze(1)


def get_worlds_initial_properties(worlds: list[World]):
    initial_props = []
    for world in worlds:
        initial_props.append(
            world.get_world_initial_properties()
        )

    return initial_props

