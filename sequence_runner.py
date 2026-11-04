import yaml
from world import World
from ui import UI
from utils.utils import get_configs
from logger import logger
from runs_handler import RunsHandler


def main():

    CONFIGS_PATH = "configs.yaml"
    WORLD_CONFIGS_PATH = "world_configs.yaml"
    UI_CONFIGS_PATH =  "ui_configs.yaml"
    
    
    print("READING CONFIGS")


    world_configs = get_configs(WORLD_CONFIGS_PATH)
    ui_configs = get_configs(UI_CONFIGS_PATH)
    configs = get_configs(CONFIGS_PATH)

    logger.set_debug_mode(debug_mode=configs["DEBUG"])
    logger.set_logs_dir("logs")

    EXPERIMENT = "LSTM BRAIN_ABC_01-23-23"
    EPISODE = 60
    WORLD = 14

    RUN_PATH = f"/home/mahmoud-sayed/Desktop/Code/Python/Reinforcement Learning Projects/World Navigator/experiments/{EXPERIMENT}/episodes"
    rh = RunsHandler(RUN_PATH)
    initial_props, actions, uoms, _ = rh.load_run(EPISODE, WORLD)
    
    world = World.from_prebuilt_world(
        world_height=                               world_configs["MIN_WORLD_HEIGHT"],
        world_width=                                world_configs["MIN_WORLD_WIDTH"],
        agent_side_length=                          world_configs["AGENT_SIDE_LENGTH"],
        world_info=initial_props
        )
    world.set_sequence_of_motion(sequence_of_motion = (actions, uoms))
    
    # print("WOLRD HAS BEEN SUCCESSFULLY INSTANTIATION")
    # print("SEQUENCE: ", sequence)
    # exit(1)
    ui = UI(
        window_width=       ui_configs["WINDOW_WIDTH"],
        window_height=      ui_configs["WINDOW_HEIGHT"],
        fps=                ui_configs["FPS"],
        margin=             ui_configs["MARGIN"],
        world=              world,
        draw_sensors=       ui_configs["DRAW_SENSORS"]
    )
    
    print("MAIN LOOP")
    i = 0
    def print_reward():
        nonlocal i
        print(f"{i}: REWARD: ", _[i])
        
        i+=1
    ui.main_loop(callback=print_reward)


if __name__ == "__main__":
    main()

#! EITHER IT DOESNT STOP WHEN IT MEETS A TERMINATION CONDITION OR SOMETHING IS WRONG WITH SEQUENCE RUNNER
