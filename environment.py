import random
from torch import Tensor
import utils
import pygame as pg
from agent import Agent

class Environment:
	def __init__(self, xmin, xmax, ymin, ymax, agent: Agent, brain=None):
		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		self.width = xmax - xmin
		self.height = ymax - ymin
		self.agent = agent
		
		## WILL BE USED ONLY WHEN 
		self.brain = brain
		
		##TODO: THIS MUST BE GENERATED ON THE RUN WHILE MAKING SURE THAT THEY DON'T MAKE A LOOP
		#* self.generate_obstacles_map()
		## OBSTACLE => ([top left], (width, height))
		self.obstacles_map = [([0, 0], (30, 70)), ([80, 100], (70, 30)), ([150, 200], (90, 10)), ([300, 150], (20, 120)), ([300, 350], (40, 110))]
		self.food_side_length = 15


		self._set_obstacles_relative_to_env_coordinates()
		
		# TODO!: ENSURE IT DOESNT INTERSECTS WITH OBSTACLES
		self._set_food_pos()
		
		#TODO!: ENSURE IT DOESNT INTERSECT WITH FOOD OR OBSTACLES
		self.agent.set_agent_initial_position_wrt_env_coordinates(self)
	

		self.create_rects()

	

	def create_rects(self):
		
		self.box = pg.Rect((self.xmin, self.ymin), (self.width, self.height))
		self.food_rect = pg.Rect((self.food_pos), (self.food_side_length, self.food_side_length))
		
		self.obstacles = []
		for obstacle in self.obstacles_map:
			rect = pg.Rect(obstacle[0], obstacle[1])
			self.obstacles.append(rect)

		self.agent.create_agent_rect()


	def _set_obstacles_relative_to_env_coordinates(self):
		for i in range(len(self.obstacles_map)):
			self.obstacles_map[i][0][0] += self.xmin
			self.obstacles_map[i][0][1] += self.ymin


	def check_for_collision_with_obstacles(self, min_max_pos):
		for obstacle in self.obstacles_map:
			if self._collide(min_max_pos, obstacle):
				return True
		return False
	#TODO!: check the whole minmax and shit
	def _collide(self, A, B): #both in min_max format
		Axmin = A[0][0]
		Axmax = A[1][0]
		Aymin = A[0][1]
		Aymax = A[1][1]
		Bxmin = B[0][0]
		Bxmax = B[1][0]
		Bymin = B[0][1]
		Bymax = B[1][1]

		return (Axmin <= Bxmax) and (Axmax >= Bxmin) and (Aymin <= Bymax) and (Aymax >= Bymin)


	def _set_food_pos(self):
		while True:
			current = (random.uniform(self.xmin, self.xmax-self.food_side_length), random.uniform(self.ymin, self.ymax-self.food_side_length))
			min_max_coords = utils.pos_wh_to_min_max(current, self.food_side_length, self.food_side_length)
			if not self.check_for_collision_with_obstacles(min_max_coords):
				break
		
		self.food_pos = current

	def _move_obstacles(self):
		pass

	def _control_agent(self, action):
		# print(self.agent.current_position)
		self.agent.move_y(1)
		self.agent.move_x(1)

	def update_env_state(self, actions):
		# THE NEURAL NETWORK AND THE ENVIRONMENT CONTROL THIS FUNCTION
		self._control_agent(actions)
		self._move_obstacles()


	def get_env_state_vector(self):

		#####################################################################################################
		##*************************************************************************************************##
		####********************************************************************************************#####
		## TODO: COMPUTE THE DISTANCES TO WALLS THROUGH UTILS, GIVE AGENT POS AND WORLD DIMS, ...          ##		  
		## TODO: ... AND REUSE IT FROM THERE WHILE COMPUTING THE DISTANCE BETWEEN THE POINT AND THE SQUARE ##
		####********************************************************************************************#####
		##*************************************************************************************************##
		#####################################################################################################
		
		agent_position = self.agent.current_position
		agent_x = agent_position[0] - self.xmin
		agent_y = agent_position[1] - self.ymin
		
		distance_to_right_wall = self.width - agent_x
		distance_to_left_wall = agent_x
		distance_to_upper_wall = -agent_y
		distance_to_lower_wall =   self.height + agent_y 

		obstacle_1_xy = self.obstacles[0][0]
		obstacle_1_wh = self.obstacles[0][1]
		obstacle_2_xy = self.obstacles[1][0]
		obstacle_2_wh = self.obstacles[1][1]
		obstacle_3_xy = self.obstacles[2][0]
		obstacle_3_wh = self.obstacles[2][1]
		obstacle_4_xy = self.obstacles[3][0]
		obstacle_4_wh = self.obstacles[3][1]

		magnitude = (self.width**2 + self.height**2)**0.5 ##! CHANGE THIS
		
		state_vector = [
			agent_x,
			agent_y,
			distance_to_left_wall, 
			distance_to_right_wall, 
			distance_to_upper_wall, 
			distance_to_lower_wall,
			obstacle_1_xy[0],
			obstacle_1_xy[1],
			obstacle_1_wh[0],
			obstacle_1_wh[1],
			obstacle_2_xy[0],
			obstacle_2_xy[1],
			obstacle_2_wh[0],
			obstacle_2_wh[1],
			obstacle_3_xy[0],
			obstacle_3_xy[1],
			obstacle_3_wh[0],
			obstacle_3_wh[1],
			obstacle_4_xy[0],
			obstacle_4_xy[1],
			obstacle_4_wh[0],
			obstacle_4_wh[1],
			self.food_pos[0],
			self.food_pos[1]
		]

		state_vector_normalized = [state_vector[i]/magnitude for i in range(len(state_vector))]

		return Tensor(state_vector_normalized)


	def evaluate_state(self):
		pass