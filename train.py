from utils import utils
from torch.optim import Adam
from logger import Logger
from datetime import datetime
from world import World, world_utils
import os
from brains import LSTMBrain
from checkpoints_handler import CheckpointsHandler


def train(world_configs: dict, network_configs: dict, training_configs: dict, configs: dict):
    
    
    

    EPISODES = training_configs["EPISODES"]
    LEARNING_RATE = training_configs["LEARNING_RATE"]
    GAMME = training_configs["GAMMA"]
    NUMBER_OF_WORLDS = training_configs["NUM_OF_WORLDS"]


    model = LSTMBrain(network_configs["LATENT_SIZE"], network_configs["N_LAYERS"], network_configs["TEMPERATURE"])
    

    actions = [[0, 0] for i in range(NUMBER_OF_WORLDS)]
    uoms = [1 for i in range(NUMBER_OF_WORLDS)]    

    for i in range(EPISODES):
        worlds = world_utils.create_worlds(
            num_worlds=                                 NUMBER_OF_WORLDS,
            worlds_height=                              world_configs["WORLD_HEIGHT"],
            worlds_width=                               world_configs["WORLD_WIDTH"],
            min_obstacles=                              world_configs["MIN_OBSTACLES"],
            max_obstacles=                              world_configs["MAX_OBSTACLES"],
            minimum_distance_between_agent_and_goal=    world_configs["MINIMUM_DISTANCE_BETWEEN_AGENT_AND_GOAL"],
            knowledge_type=                             world_configs["KNOWLEDGE_TYPE"],
            agent_side_length=                          world_configs["AGENT_SIDE_LENGTH"],
            obstacle_constraints_shorter_side=          world_configs["OBSTACLE_CONSTRAINTS_SHORTER_SIDE"],
            obstacle_constraints_longer_side=           world_configs["OBSTACLE_CONSTRAINTS_LONGER_SIDE"],
            min_unit_of_motion=                         world_configs["MIN_UNIT_OF_MOTION"],
            max_unit_of_motion=                         world_configs["MIN_UNIT_OF_MOTION"],
            passage_space_factor=                       world_configs["PASSAGE_SPACE_FACTOR"]
        )

        next_states = world_utils.move_worlds_to_next_state(worlds, actions, uoms)
        ## APPLY PADDING WITH MAX OBSTACLES
        agents_states, goals_states, obstacles_states = None, None, None # some function that will extract these values from the `next_state` object 
        #LOGITS RATHER THAN ACTIONS
        actions, uoms = model(agents_states, goals_states, obstacles_states)

        
        pass
    
    
    pass

def create_training_environment(base_dir, model_name, experiment_notes):
    timestamp = datetime.now().strftime('%H-%M-%S')
    experiment_name = f"{model_name}_{experiment_notes}_{timestamp}"
    env_path = os.path.join(base_dir, experiment_name)
    logs_path = os.path.join(env_path, "logs")
    ckpts_path = os.path.join(env_path, "checkpoints")
    runs_path = os.path.join(env_path, "episodes")

    os.mkdir(path=env_path)
    os.mkdir(path=logs_path)
    os.mkdir(path=ckpts_path)
    os.mkdir(path=runs_path)

    ##      - LOGS: 
    ##      - MODEL-CHECKPOINT:
    ##          - 1
    ##          - 2
    ##          - ...
    ##      - EPISODES DATA:
    ##          - 1:
    ##              - SIMULATION1
    ##              - SIMULATION2
    ##              - ...
    ##          - 2:

    return env_path, logs_path, ckpts_path, runs_path, timestamp




def main():


    WORLD_CONFIGS_PATH =    "world_configs.yaml"
    NETOWRK_CONFIGS_PATH =  "brains/configs/network_configs/lstm_brain_configs.yaml"
    TRAINING_CONFIGS_PATH = "brains/configs/training_configs/lstm_brain_configs.yaml"
    CONFIGS_PATH =          "configs.yaml"

    world_configs =     utils.get_configs(WORLD_CONFIGS_PATH)
    network_configs =   utils.get_configs(NETOWRK_CONFIGS_PATH)
    training_configs =  utils.get_configs(TRAINING_CONFIGS_PATH)
    configs =           utils.get_configs(CONFIGS_PATH)
    
    env_path, logs_path, ckpts_path, runs_path, timestamp = create_training_environment(base_dir= configs["TRAINING_BASE"], model_name=configs["MODEL_NAME"], experiment_notes= configs["EXPERIMENTS_NOTES"])
    logger = Logger(debug_mode=True, logs_dir_path=logs_path)
    logger.log(f"Run started at: {timestamp}")
    logger.log(f"Run Environment created at: {env_path}")
    configs["ENV_PATH"] = env_path
    configs["LOGS_PATH"] = logs_path
    configs["CKPTS_PATH"] = ckpts_path
    configs["RUNS_PATH"] = runs_path

    train(world_configs, network_configs, training_configs, configs)




if __name__ == "__main__":
    main()