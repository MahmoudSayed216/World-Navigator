from brains import LSTMBrain
import torch



def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)




agent_state = torch.rand(size=(32, 3))
goal_state = torch.rand(size=(32, 2))
obstacles_state = torch.rand(size=(32, 15, 5))

model = LSTMBrain(latent_size=48,temperature=2)


output = model(agent_state, goal_state, obstacles_state)


count_params(model)