import torch
import torch.nn as nn




class LSTMBrain(nn.Module):
    def __init__(self, latent_size: int = 64, n_layers: int = 2, temperature: torch.Tensor = torch.scalar_tensor(s=1)):
        super().__init__()

        self.T = temperature
        # agent processor
        self.agent_processor1 = nn.Linear(in_features=3, out_features=64)
        self.agent_bn1 = nn.BatchNorm1d(num_features=64)
        self.agent_non_linearity1 = nn.ReLU()
        self.agent_processor2 = nn.Linear(in_features=64, out_features=latent_size)
        self.agent_bn2 = nn.BatchNorm1d(num_features=latent_size)
        self.agent_non_linearity2 = nn.ReLU()


        # goal processor
        self.goal_processor1 = nn.Linear(in_features=2, out_features=64)
        self.goal_bn1 = nn.BatchNorm1d(num_features=64)
        self.goal_non_linearity1 = nn.ReLU()
        self.goal_processor2 = nn.Linear(in_features=64, out_features= latent_size)
        self.goal_bn2 = nn.BatchNorm1d(num_features=latent_size)
        self.goal_non_linearity2 = nn.ReLU()

        # obstacles processor
        self.obstacles_processor1 = nn.LSTM(input_size=5, hidden_size=latent_size, num_layers=n_layers, batch_first=True)
        self.obstacles_bn1 = nn.BatchNorm1d(num_features=latent_size)
        self.obstacles_non_linearity1 = nn.ReLU()

        # world processor
        self.world_processor1 = nn.Linear(in_features=latent_size*3, out_features=latent_size)
        self.world_bn1 = nn.BatchNorm1d(num_features=latent_size)
        self.world_non_linearity1 = nn.ReLU()
        self.world_processor2 = nn.Linear(in_features=latent_size, out_features=32)
        self.world_bn2 = nn.BatchNorm1d(num_features=32)
        self.world_non_linearity2 = nn.ReLU()

        self.direction_output = nn.Linear(in_features=32, out_features=4)
        self.unit_of_motion_output = nn.Linear(in_features=32, out_features=1)

        self.direction_smx = nn.Softmax(dim=1)
        self.unit_of_motion_sgmd = nn.Sigmoid()






    def forward(self, agent_state, goal_state, obstacles_state):
        # agent state processing
        agent_state = self.agent_processor1(agent_state)
        agent_state = self.agent_bn1(agent_state)
        agent_state = self.agent_non_linearity1(agent_state)
        agent_state = self.agent_processor2(agent_state)
        agent_state = self.agent_bn2(agent_state)
        agent_state = self.agent_non_linearity2(agent_state)


        # goal state processing
        goal_state = self.goal_processor1(goal_state)
        goal_state = self.goal_bn1(goal_state)
        goal_state = self.goal_non_linearity1(goal_state)
        goal_state = self.goal_processor2(goal_state)
        goal_state = self.goal_bn2(goal_state)
        goal_state = self.goal_non_linearity2(goal_state)


        # obstacles state processing
        output, (obstacles_state, c_n) = self.obstacles_processor1(obstacles_state)
        obstacles_state = self.obstacles_bn1(obstacles_state[-1])
        obstacles_state = self.obstacles_non_linearity1(obstacles_state)


        # world state processing
        world_state = torch.cat(tensors=(agent_state, goal_state, obstacles_state), dim=1)

        world_state = self.world_processor1(world_state)
        world_state = self.world_bn1(world_state)
        world_state = self.world_non_linearity1(world_state)
        world_state = self.world_processor2(world_state)
        world_state = self.world_bn2(world_state)
        world_state = self.world_non_linearity2(world_state)


        direction = self.direction_output(world_state) / self.T
        uom = self.unit_of_motion_output(world_state)

        direction_dist = self.direction_smx(direction)
        uom = self.unit_of_motion_sgmd(uom)


        return direction_dist, uom
        
        

    