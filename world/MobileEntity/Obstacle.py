from world.MobileEntity.MobileEntity import MobileEntity
from typing import Literal

class Obstacle(MobileEntity):
    def __init__(self, x_pos, y_pos, width, height, unit_of_motion, range_of_motion, axis_of_motion: Literal["x", "y"]):
        super().__init__(x_pos, y_pos, width, height, unit_of_motion)
        self.axis_of_motion = axis_of_motion
        self.calculate_motion_bounds(range_of_motion)
        self.direction = 1
    def calculate_motion_bounds(self, range_of_motion):
        if self.axis_of_motion == "x":
            self.upper_bound = self.x_pos + range_of_motion
            self.lower_bound = self.x_pos - range_of_motion
        if self.axis_of_motion == "y":
            self.upper_bound = self.y_pos + range_of_motion
            self.lower_bound = self.y_pos - range_of_motion
        

    def next_state(self):
        if self.axis_of_motion == "x":
            self.x_pos = self.x_pos + self.direction*self.unit_of_motion
            if (self.x_pos >= self.upper_bound) or (self.x_pos <= self.lower_bound):
                self.direction = self.direction*-1
        if self.axis_of_motion == "y":
            self.y_pos = self.y_pos + self.direction*self.unit_of_motion
            if (self.y_pos >= self.upper_bound) or (self.y_pos <= self.lower_bound):
                    self.direction = self.direction*-1
        
        