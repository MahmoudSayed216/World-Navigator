import pygame as pg
from colors import *
from agent import Agent
from environment import Environment


##TODO!: THE UI CLASS WILL SET THE X,Y POSITIONS FOR BOTH THE ENV AND THE NN, 
	   #!SO REMOVE X n Y FROM BOTH ENTITIES

#TODO! PROBABLY CHANGE THE UI NAME AN CHANGE IT TO THE NAME OF THE PROGRAM MAYBE?


class UI:
	def __init__(self, environment: Environment, window_w, window_h, network_architecture = [12, 30, 30, 2]):
		self.environment = environment
		pg.init()
		pg.font.init()
		self.WIN = pg.display.set_mode((window_w, window_h))
		pg.display.set_caption("Reinforcement Learning simulator")
		self.FPS = 60
		self.clock = pg.time.Clock()
		self.network_architecture = network_architecture
		self._init_objects()


	def _init_objects(self):

		##TODO: MAKE NN AN INDEPENDENT CLASS

		self.NN_rect_x = self.environment.xmin + self.environment.width +50
		self.NN_rect = pg.Rect((self.NN_rect_x, self.environment.ymin), (self.environment.width, self.environment.height))

	def draw_objects(self):
		pg.draw.rect(self.WIN, LIGHT_GRAY, self.environment.box)
		pg.draw.rect(self.WIN, SKY_BLUE, self.NN_rect)
		for rect in self.environment.obstacles:
			pg.draw.rect(self.WIN, BLACK, rect)
		pg.draw.rect(self.WIN, LIGHT_RED, self.environment.agent.agent_rect)
		pg.draw.rect(self.WIN, LIGHT_GREEN, self.environment.food_rect)

	def draw_nn(self):
		# TODO: TAKES THE PYTORCH NETWORK OBJECT AS INPUT, AND CONVERTS IT TO A BUNCH OF CIRCLES AND LINES
		# TODO: functionize this
		x_factor = (self.environment.width - 50) / len(self.network_architecture)-1
		x_s = [self.NN_rect_x + i*x_factor + 50 for i in range(len(self.network_architecture))]
		for x, n_neurons in zip(x_s, self.network_architecture):
				y_factor = (self.environment.height-50) / (n_neurons - 1)
				for i in range(n_neurons):
					pg.draw.circle(self.WIN, WHITE, (x, i*y_factor+ 25 + self.environment.ymin), 5)
	
	def draw_environment(self):
		self.WIN.fill(DARK_GRAY)
		self.draw_objects()
		self.draw_nn()
		pg.display.update()
	
	#TODO: _update_agent_pos will be controlled by the neural network

	def main_loop(self):
		run = True
		while run:
			self.clock.tick(self.FPS)
			self.draw_environment()
			for event in pg.event.get():
				if event.type == pg.QUIT:
					run = False
			#TODO: update_agent_pos will be controlled by the neural network		
			self.environment.update_env_state('')
			
			
		
		pg.quit()
