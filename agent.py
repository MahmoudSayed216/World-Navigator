# from numpy import random
import utils
import random
import pygame as pg

##! THE BRAIN SHOULD PROBABLY BE GIVEN TO THE AGENT NOT THE ENVIRONMENT

class Agent:
	def __init__(self, width, height):
		self.current_position = None
		self.width = width
		self.height = height
		self.velocity = 1
		self.DISTANCE_THRESHOLD = 200

	def set_agent_initial_position_wrt_env_coordinates(self, environment):
		while True:
			current_pos = (random.uniform(environment.xmin, environment.xmax - self.width), random.uniform(environment.ymin, environment.ymax - self.height))
			min_max_coords = utils.pos_wh_to_min_max(current_pos, self.width, self.height)
			distance= utils.euclidean_distance(current_pos, environment.food_pos)
			print(distance)
			if not environment.check_for_collision_with_obstacles(min_max_coords) and  distance > self.DISTANCE_THRESHOLD:
				break
		self.current_position = (environment.xmin, environment.ymin)
	
	def create_agent_rect(self):
		self.agent_rect = pg.Rect(self.current_position, (self.width, self.height))

	def move_y(self, v):
		self.current_position = (self.current_position[0], self.current_position[1]-v)
		
		self.agent_rect.move_ip(0, v)

	def move_x(self, v):
		self.current_position = (self.current_position[0]+v, self.current_position[1])
		self.agent_rect.move_ip(v,0)


	
		


	##TODO! IT SHOULD BE GIVEN THE OPTION TO MOVE BY FRACTIONS