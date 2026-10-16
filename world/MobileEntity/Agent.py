from world.MobileEntity.MobileEntity import MobileEntity
from torch import Tensor

class Agent(MobileEntity):
    def __init__(self, x_pos, y_pos, width, height, unit_of_motion, n_sensors):
        super().__init__(x_pos, y_pos, width, height, unit_of_motion)
        self.n_sensors = n_sensors
        self.create_sensors()

    def create_sensors(self):
        # create a vertical line that starts from the center of the agent, and other lines with equal angles from each other.
        #
        pass 


    #The action Tensor has one of 4 possible forms -> (1, 0), (-1, 0), (0, 1), (0, -1)
    def next_state(self, action: Tensor):
        self.x_pos += action[0]*self.unit_of_motion
        self.y_pos += action[1]*self.unit_of_motion