import pygame as pg
from colors import *
import torch
from world import World
# DRAWS BOTH WORLD AND NETWORK

class UI:
    def __init__(self, window_width, window_height, margin, fps, world: World, brain = None, draw_sensors= False):
        pg.init()
        pg.font.init()
        self.WIN = pg.display.set_mode((window_width, window_height))
        pg.display.set_caption("Reinforcement Learning simulator")
        self.margin = margin
        self.FPS = fps
        self.clock = pg.time.Clock()
        self.world = world
        self.brain = brain
        self.draw_sensors = draw_sensors
        # self.create_world_objects()

    def create_world_objects(self):
        self.create_rects()
        if self.draw_sensors:
            self.create_sensors_lines() # MUST CONSIDER KNOWLEDGE TYPE

    def create_sensors_lines(self):
        state = self.world.get_current_state() #0 -> agent, 1 -> goal, ... -> obs
        self.sensors_lines = []
        agent_x_c = state[0] + state[2]/2 + self.margin
        agent_y_c = state[1] + state[3]/2 + self.margin

        for i in range(8, len(state)-1, 4):
            x1, y1 = state[i]+self.margin, state[i+1]+self.margin
            x2, y2 = x1+state[i+2], y1+state[i+3]
            self.sensors_lines.append(((x1, y1), (agent_x_c, agent_y_c)))
            self.sensors_lines.append(((x2, y2), (agent_x_c, agent_y_c)))


    def create_rects(self):
        self.world_rect = pg.Rect((self.margin, self.margin), (self.world.world_width, self.world.world_height))
        agx, agy = self.world.agent.get_coordinates()
        self.agent_rect = pg.Rect((agx+self.margin, agy+self.margin), self.world.agent.get_dims())
        glx, gly = self.world.goal.get_coordinates()
        self.goal_rect = pg.Rect((glx+self.margin, gly+self.margin), self.world.goal.get_dims())
        self.obstacles_rects = []
        for obstacle in self.world.obstacles:
            obx, oby = obstacle.get_coordinates()
            ob_rect = pg.Rect((obx+self.margin, oby+self.margin), obstacle.get_dims())
            self.obstacles_rects.append(ob_rect)

        

    def draw_world_objects(self):
        pg.draw.rect(self.WIN, color=MINT, rect=self.world_rect)
        pg.draw.rect(self.WIN, color=SAPPHIRE, rect=self.agent_rect)
        pg.draw.rect(self.WIN, color=CORAL, rect=self.goal_rect)
        for obstacle_rect in self.obstacles_rects:
            pg.draw.rect(self.WIN, color=SLATE_BLUE, rect= obstacle_rect)
        if self.draw_sensors:
            for line_pair in self.sensors_lines:
                pg.draw.line(self.WIN, color=DUSTY_ROSE, start_pos=line_pair[0], end_pos=line_pair[1], width=1)

            
        


    def draw_ui(self):
        self.WIN.fill(WHITE)
        self.draw_world_objects()
        # self.draw_nn()
        pg.display.update()
	


    def main_loop(self):
        run = True
        while run:
            self.clock.tick(self.FPS)
            self.create_world_objects()
            self.draw_ui()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False
            #? I THINK THESE 2 LAYERS SHOULDN'T BE ABLE TO SEE EACH OTHER
            print("next state: ", self.world.next_state([1, 0], 1))
            
			
			
		
        pg.quit()