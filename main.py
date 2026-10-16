# CREATE A FILE CALLED SIMULATE.PY THAT:
    # READS THE NETWORK
    # IMPORTS THE WORLD
    # INSTANTIATES A WORLD
    # RUNS FULL SIMULATION AND GENERATES A SEQUENCE
    # VISUALIZES THE SEQUENCE   

#! THERE'S CLEARLY A PROBLEM WITH THE INITIAL DISTANCE BETWEEN THE GOAL AND THE AGENT
#! THERE'S CLEARLY A PROBLEM WITH THE DIMENSIONS OF THE OBSTACLES AND AS A RESULT, THEY ALL MOVE IN THE SAME DIRECTION

import yaml
from world import World
from ui import UI
from utils.utils import get_configs




def main():

    CONFIGS_PATH = "configs.yaml"
    WORLD_CONFIGS_PATH = "world_configs.yaml"
    UI_CONFIGS_PATH =  "ui_configs.yaml"
    
    
    print("READING CONFIGS")


    world_configs = get_configs(WORLD_CONFIGS_PATH)
    ui_configs = get_configs(UI_CONFIGS_PATH)
    configs = get_configs(CONFIGS_PATH)

    world = World(
        world_height=                               world_configs["WORLD_HEIGHT"],
        world_width=                                world_configs["WORLD_WIDTH"],
        n_obstacles=                                world_configs["MIN_OBSTACLES"],
        minimum_distance_between_agent_and_goal=    world_configs["MINIMUM_DISTANCE_BETWEEN_AGENT_AND_GOAL"],
        knowledge_type=                             world_configs["KNOWLEDGE"],
        agent_side_length=                          world_configs["AGENT_SIDE_LENGTH"],
        obstacle_constraints_shorter_side=          world_configs["OBSTACLE_CONSTRAINTS_SHORTER_SIDE"],
        obstacle_constraints_longer_side=           world_configs["OBSTACLE_CONSTRAINTS_LONGER_SIDE"],
        unit_of_motion=                             world_configs["MIN_UNIT_OF_MOTION"],
        passage_space_factor=                       world_configs["PASSAGE_SPACE_FACTOR"],
    )

    print("WOLRD HAS BEEN SUCCESSFULLY INSTANTIATION")


    ui = UI(
        window_width=       ui_configs["WINDOW_WIDTH"],
        window_height=      ui_configs["WINDOW_HEIGHT"],
        fps=                ui_configs["FPS"],
        margin=             ui_configs["MARGIN"],
        world=              world,
        draw_sensors=         ui_configs["DRAW_SENSORS"]
    )
    
    print("MAIN LOOP")
    ui.main_loop()


if __name__ == "__main__":
    main()