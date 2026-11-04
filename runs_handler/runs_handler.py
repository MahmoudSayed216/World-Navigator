import pickle as pkl
import os


class RunsHandler:
    """
    Saves the per-world histories of each training episode.

    Layout:
        runs_dir_path/
            1/
                0.pkl      <- world 0 of episode 1
                1.pkl
                ...
            2/
                ...
    """

    def __init__(self, runs_dir_path):
        self.runs_dir_path = runs_dir_path
        self.current_episode = 0   # int counter
        self.current_dir = ""      # path of the current episode's folder
        os.makedirs(self.runs_dir_path, exist_ok=True)

    def _create_dir_for_new_episode(self):
        self.current_episode += 1
        self.current_dir = os.path.join(self.runs_dir_path, str(self.current_episode))
        os.makedirs(self.current_dir, exist_ok=True)

    def _save_episode(self, initial_props, actions, uoms, rewards):
        # each argument is indexed by world: actions[w] is the full history of world w
        n_worlds = len(actions)
        assert len(uoms) == n_worlds and len(rewards) == n_worlds, \
            "actions, uoms and rewards must have one entry per world"

        for world_idx in range(n_worlds):
            world_state_dict = {
                "initial_props":initial_props[world_idx],
                "actions": actions[world_idx],
                "uoms": uoms[world_idx],
                "rewards": rewards[world_idx],
            }
            world_path = os.path.join(self.current_dir, f"{world_idx}.pkl")
            with open(world_path, "wb") as file:
                pkl.dump(world_state_dict, file)

    def step(self,initial_props, actions, uoms, rewards):
        self._create_dir_for_new_episode()
        self._save_episode(initial_props, actions, uoms, rewards)

    def load_run(self, episode, world_number):
        world_path = os.path.join(self.runs_dir_path, str(episode), f"{world_number}.pkl")
        with open(world_path, "rb") as file:
            run = pkl.load(file)

        return run["initial_props"], run["actions"], run["uoms"], run["rewards"]