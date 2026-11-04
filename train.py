from utils import utils
from torch.optim import Adam
from logger import logger
from datetime import datetime
from world import World, world_utils
import os
from brains import DeepSetsBrain
from checkpoints_handler import CheckpointsHandler
from runs_handler import RunsHandler
from torch.distributions import Categorical
from rewards import BasicReward
import torch
# THE PROJECT IN ITS CURRENT STATE DOESNT SUPPORT CONTINUOUING FROM A PREVIOUS RUN
# ESPECIALLY THE RUNS HANDELR AND THE TRAIN FUNCTION
# ITS ESSENTIALLY UNNEEDED




def train(world_configs: dict, network_configs: dict, training_configs: dict, configs: dict, reward_configs: dict):
    
    
    logger.log("Initializing training environment")

    EPISODES = training_configs["EPISODES"]
    LEARNING_RATE = training_configs["LEARNING_RATE"]
    GAMMA = training_configs["GAMMA"]
    NUMBER_OF_WORLDS = training_configs["NUM_OF_WORLDS"]
    MAX_STEPS = world_configs["MAX_STEPS"]
    temperature = network_configs["TEMPERATURE"]
    TEMPERATURE_INCREASE_FACTOR = network_configs["TEMPERATURE_INCREASE_FACTOR"]
    MAX_OBSTACLES = world_configs["MAX_OBSTACLES"]
    WORLD_WIDTH = world_configs["MIN_WORLD_WIDTH"]
    WORLD_HEIGHT = world_configs["MIN_WORLD_HEIGHT"]
    MAX_UNIT_OF_MOTION = world_configs["MAX_UNIT_OF_MOTION"]
    # model = LSTMBrain(, network_configs["N_LAYERS"])
    model = DeepSetsBrain(network_configs["LATENT_SIZE"], n_layers=6, )
    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)


    THETA = reward_configs["THETA"]
    BETA = reward_configs["BETA"]
    C = reward_configs["C"]
    K = reward_configs["K"]
    
    reward_f = BasicReward(THETA, BETA, C, K)
    ckpt_handler = CheckpointsHandler(save_every= training_configs["SAVE_EVERY"], increasing_metric=True, output_path=configs["CKPTS_PATH"])
    runs_handler = RunsHandler(runs_dir_path=configs["RUNS_PATH"])

    j = 0

    logger.log(f"""training started with paramter set:
                    Model Name: {network_configs["MODEL_NAME"]}
                    Model Parameter Count: {utils.count_params(model)}
                    Model Architecture: \n{model}
                    Hyper Parameters: 
                        Episodes: {EPISODES}
                        Learning Rate: {LEARNING_RATE}
                        Discounting GAMMA: {GAMMA}
                        Number of Worlds: {NUMBER_OF_WORLDS}
                        Max Steps Per Episode: {MAX_STEPS}
                        """)

    for i in range(1, EPISODES+1):
        optimizer.zero_grad()    
        logger.log("Creating worlds ...")
        worlds = world_utils.create_worlds(
            num_worlds=                                 NUMBER_OF_WORLDS,
            worlds_height=                              WORLD_WIDTH,
            worlds_width=                               WORLD_HEIGHT,
            min_obstacles=                              world_configs["MIN_OBSTACLES"],
            max_obstacles=                              world_configs["MAX_OBSTACLES"],
            minimum_distance_between_agent_and_goal=    world_configs["MINIMUM_DISTANCE_BETWEEN_AGENT_AND_GOAL_LB"],
            maximum_distance_between_agent_and_goal=    world_configs["MINIMUM_DISTANCE_BETWEEN_AGENT_AND_GOAL_UB"],
            knowledge_type=                             world_configs["KNOWLEDGE_TYPE"],
            agent_side_length=                          world_configs["AGENT_SIDE_LENGTH"],
            obstacle_constraints_shorter_side=          world_configs["OBSTACLE_CONSTRAINTS_SHORTER_SIDE"],
            obstacle_constraints_longer_side=           world_configs["OBSTACLE_CONSTRAINTS_LONGER_SIDE"],
            min_unit_of_motion=                         world_configs["MIN_UNIT_OF_MOTION"],
            max_unit_of_motion=                         MAX_UNIT_OF_MOTION,
            passage_space_factor=                       world_configs["PASSAGE_SPACE_FACTOR"]
        )
        # logger.log(f"Episode:  {str(i).rjust(3)} / {EPISODES}")
        # print("NEW EPOCH")
        world_initial_properties = world_utils.get_worlds_initial_properties(worlds)
        actions = [[] for _ in range(NUMBER_OF_WORLDS)]
        uoms = [[] for _ in range(NUMBER_OF_WORLDS)]
        rewards = [[] for _ in range(NUMBER_OF_WORLDS)]
        log_probs = []                                     # keeps the graph, used for the loss
        logger.log(f"Episode:  {str(i).rjust(3)} / {EPISODES}")
        temperature = temperature * TEMPERATURE_INCREASE_FACTOR
        active_masks = []
        active_mask = torch.ones(NUMBER_OF_WORLDS)
        for step in range(1, MAX_STEPS+1):
            # if step%500 == 0:
                # logger.log(f"Episode:  {str(i).rjust(3)} / {EPISODES},  Step: {str(step).rjust(4)} / {MAX_STEPS}")
            
            agents_states, goals_states, obstacles_states, _ = world_utils.process_states_vector(
                world_utils.get_worlds_current_state(worlds), MAX_OBSTACLES, WORLD_WIDTH, WORLD_HEIGHT, MAX_UNIT_OF_MOTION)

            actions_logits, uom_logits = model(agents_states, goals_states, obstacles_states)
            action_dist = Categorical(logits=actions_logits / temperature)
            uom_dist = Categorical(logits=uom_logits)

            a = action_dist.sample()
            u = uom_dist.sample()                          # + 1 if uom must be >= 1
            log_probs.append(action_dist.log_prob(a) + uom_dist.log_prob(u))   # (B,), has grad
            active_masks.append(active_mask.clone())

            new_states = world_utils.move_worlds_to_next_state(worlds, a, u+1)
            n_agents, n_goals, _, n_term = world_utils.process_states_vector(new_states, MAX_OBSTACLES, WORLD_WIDTH, WORLD_HEIGHT, MAX_UNIT_OF_MOTION)
            active_mask *= (n_term == 0).float()
            goal_xy = n_goals + n_agents[:, :2]

            r = reward_f.reward(agents_states, n_agents, goal_xy, n_term)      # (B,), no grad

            for w in range(NUMBER_OF_WORLDS):
                actions[w].append(a[w].item())
                uoms[w].append(u[w].item()+1)
                rewards[w].append(r[w].item())
        
        
        log_probs_t = torch.stack(log_probs)               # (T, B), has grad
        rewards_t   = torch.tensor(rewards).T              # rewards is B lists of T floats -> (T, B)
        active_masks_t = torch.stack(active_masks)

        returns = torch.zeros_like(rewards_t)
        G = torch.zeros(NUMBER_OF_WORLDS)
        for t in reversed(range(MAX_STEPS)):
            G = rewards_t[t] + GAMMA * G
            returns[t] = G

        valid_returns = returns[active_masks_t.bool()]

        adv = torch.zeros_like(returns)

        adv[active_masks_t.bool()] = (
            valid_returns - valid_returns.mean()
        ) / (valid_returns.std() + 1e-8)


        loss = -(log_probs_t * adv * active_masks_t).sum() / active_masks_t.sum().clamp_min(1)
        loss.backward()
        optimizer.step()
        mean_total_reward = rewards_t.sum(0).mean().item()   # sum over steps -> (B,), mean over worlds
        
        total_rewards = rewards_t.sum(0)  # Total reward per world
        best_world_idx = total_rewards.argmax().item()
        best_world_reward = total_rewards[best_world_idx].item()
        
        logger.log(
           f"LOSS: {loss.item()} | "
            f"MEAN TOTAL REWARD: {mean_total_reward} | "
            f"BEST WORLD: {best_world_idx} | "
            f"BEST REWARD: {best_world_reward}"
        )
        runs_handler.step(initial_props= world_initial_properties, actions= actions, uoms=uoms, rewards=rewards)
        #! wrong layer, ckpt doesn't belong here, it's formed within the .step function after taking all ckpt relevant content
        ckpt_handler.step(metric_val=0, current_epoch=i, ckpt=model.state_dict())


        
        pass
    
    
    pass

def create_training_environment(base_dir, model_name, experiment_notes):
    timestamp = datetime.now().strftime('%H-%M-%S')
    experiment_name = f"{model_name}_{experiment_notes}_{timestamp}"
    env_path = os.path.join(base_dir, experiment_name)
    logs_path = os.path.join(env_path, "logs")
    ckpts_path = os.path.join(env_path, "checkpoints")
    runs_path = os.path.join(env_path, "episodes")
    os.makedirs("./experiments", exist_ok=True)
    os.mkdir(path=env_path)
    os.mkdir(path=logs_path)
    os.mkdir(path=ckpts_path)
    os.mkdir(path=runs_path)

    ##      - LOGS: 
    ##          - last.pth
    ##          - last.pth
    ##          - last.pth
    ##          - last.pth
    ##      - MODEL-CHECKPOINT:
    ##          - last.pth
    ##          - best.pth
    ##      - EPISODES DATA:
    ##          - 1:
    ##              - SIMULATION1
    ##              - SIMULATION2
    ##              - ...
    ##          - 2:

    return env_path, logs_path, ckpts_path, runs_path, timestamp




def main():


    WORLD_CONFIGS_PATH =    "world_configs.yaml"
    NETOWRK_CONFIGS_PATH =  "brains/configs/network_configs/deep_sets_brain_configs.yaml"
    TRAINING_CONFIGS_PATH = "brains/configs/training_configs/deep_sets_brain_training_configs.yaml"
    REWARD_CONFIGS_PATH = "reward_configs.yaml"
    CONFIGS_PATH =          "configs.yaml"

    world_configs =     utils.get_configs(WORLD_CONFIGS_PATH)
    network_configs =   utils.get_configs(NETOWRK_CONFIGS_PATH)
    training_configs =  utils.get_configs(TRAINING_CONFIGS_PATH)
    reward_configs = utils.get_configs(REWARD_CONFIGS_PATH)
    configs =           utils.get_configs(CONFIGS_PATH)
    
    env_path, logs_path, ckpts_path, runs_path, timestamp = create_training_environment(base_dir= configs["TRAINING_BASE"], model_name=configs["MODEL_NAME"], experiment_notes= configs["EXPERIMENT_NOTES"], )

    logger.set_debug_mode(debug_mode=configs["DEBUG"])
    logger.set_logs_dir(logs_path)

    logger.log(f"Run started at: {timestamp}")
    logger.log(f"Run Environment created at: {env_path}")
    
    configs["ENV_PATH"] = env_path
    configs["LOGS_PATH"] = logs_path
    configs["CKPTS_PATH"] = ckpts_path
    configs["RUNS_PATH"] = runs_path

    train(world_configs, network_configs, training_configs, configs, reward_configs)




if __name__ == "__main__":
    main()