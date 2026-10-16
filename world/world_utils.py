from world import World
import random

def create_worlds(num_worlds, worlds_height, worlds_width, min_obstacles, max_obstacles, minimum_distance_between_agent_and_goal,
                  knowledge_type, agent_side_length, obstacle_constraints_shorter_side, obstacle_constraints_longer_side,
                  min_unit_of_motion, max_unit_of_motion, passage_space_factor):
    
    worlds = []
    for i in range(num_worlds):
        n_obst = int(random.uniform(min_obstacles, max_obstacles)+1)
        uom = int(random.uniform(min_unit_of_motion, max_unit_of_motion)+1)
        world = World(
            world_height=                               worlds_height,
            world_width=                                worlds_width,
            n_obstacles=                                n_obst,
            minimum_distance_between_agent_and_goal=    minimum_distance_between_agent_and_goal,
            knowledge_type=                             knowledge_type,
            agent_side_length=                          agent_side_length,
            obstacle_constraints_shorter_side=          obstacle_constraints_shorter_side,
            obstacle_constraints_longer_side=           obstacle_constraints_longer_side,
            unit_of_motion=                             uom,
            passage_space_factor=                       passage_space_factor
        )

        worlds.append(world)

    return world



def move_worlds_to_next_state(worlds: list[World], actions, uoms):
    new_states = []
    for world, action in enumerate((worlds, actions)):
        state = world.next_state(action)
        new_states.append(state)

    return new_states