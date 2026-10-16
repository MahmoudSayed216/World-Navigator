from abc import ABC, abstractmethod


class MobileEntity(ABC):
    def __init__(self, x_pos, y_pos, width, height, unit_of_motion):
        super().__init__()
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.width = width
        self.height = height
        self.unit_of_motion = unit_of_motion

    @abstractmethod
    def next_state(self):
        pass

    def get_entity_spatial_attributes(self):
        return self.x_pos, self.y_pos, self.width, self.height

    def get_coordinates(self):
        return self.x_pos, self.y_pos

    def get_dims(self):
        return self.width, self.height
    def set_unit_of_motion(self, unit_of_motion):
        self.unit_of_motion = unit_of_motion