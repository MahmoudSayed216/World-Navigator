from world.MobileEntity import Obstacle, Agent
import random
from utils import utils
from logger import logger



class World:
    ACTIONS_MAP = {
        0: [0, 1], #down 
        1: [0, -1], #up
        2: [1, 0],
        3: [-1, 0]
    }        
    def __init__(
            self,
            world_width, 
            world_height, 
            n_obstacles, 
            minimum_distance_between_agent_and_goal, 
            knowledge_type, 
            agent_side_length,
            obstacle_constraints_shorter_side,
            obstacle_constraints_longer_side,
            unit_of_motion,
            passage_space_factor
            ):

        self.world_width = world_width
        self.world_height = world_height
        self.n_obstacles = n_obstacles
        self.minimum_distance_between_agent_and_goal = minimum_distance_between_agent_and_goal
        self.knowledge_type = knowledge_type
        self.agent_side_length = agent_side_length
        self.obstacle_constraints_shorter_side = obstacle_constraints_shorter_side
        self.obstacle_constraints_longer_side = obstacle_constraints_longer_side
        self.unit_of_motion = unit_of_motion
        self.passage_space_factor = passage_space_factor
        self.terminated = 0

        self.create_agent()
        self.create_goal() #ENSURE A MINIMUM DISTANCE BETWEEN AGENT AND goal
        self.create_obstacles() #ENSURE TRAJECTORY WON'T INTERFER WITH goal

        # logger.log(f"World created with properties: \n{self.world_info()}", end="\n")


    def create_agent(self):
        x, y = int(random.uniform(self.agent_side_length, self.world_width-self.agent_side_length)), int(random.uniform(self.agent_side_length, self.world_height-self.agent_side_length))
        self.agent = Agent(x, y, self.agent_side_length, self.agent_side_length, self.unit_of_motion, 0)
        # self.agent = Agent(50, 10, self.agent_side_length, self.agent_side_length, self.unit_of_motion, 0)

    def world_info(self):
        info_str = ""
        info_str += f"\t\tWorld Height: {self.world_height}\n"
        info_str += f"\t\tWorld Width: {self.world_width}\n"
        info_str += f"\t\tNumber of Obstacles: {self.n_obstacles}\n"
        info_str += f"\t\tMin Distance Between Agent and Goal: {self.minimum_distance_between_agent_and_goal}\n"
        info_str += f"\t\tAgent: \n{self.agent.entity_info()}"
        info_str += f"\t\tGoal: \n{self.goal.entity_info()}"

        return info_str


    
    def create_goal(self):
        
        while True:

            x, y = int(random.uniform(0, self.world_width-self.agent_side_length)), int(random.uniform(0, self.world_height-self.agent_side_length))
            distance = utils.euclidean_distance(*self.agent.get_coordinates(), x, y)
            
            if distance >= self.minimum_distance_between_agent_and_goal:
                break

        
        self.goal = Agent(x, y, self.agent_side_length, self.agent_side_length, self.unit_of_motion, 0)
        # self.goal = Agent(500, 10, self.agent_side_length, self.agent_side_length, self.unit_of_motion, 0)
        

    def create_obstacles(self):
        self.obstacles = []
        for i in range(self.n_obstacles):
            while True:
                w, h, range_of_motion, axis, x, y = self.create_obstacle_spatial_attributes()

                if self.check_for_collisions(x, y, w, h, range_of_motion, axis) == False: #No Collisions
                    break
            # print(f"{i+1} obstacles have been created")
            obstacle = Obstacle(x, y, w, h, self.unit_of_motion, range_of_motion, axis)#NEEDS FURTHER VALIDATION TO ENSURE IT WONT COLLIDED WITH GOAL WHILE MOVING)
            self.obstacles.append(obstacle)



    def create_obstacle_spatial_attributes(self):
        width_favored = random.choice([0, 1])
        v1 = int(random.uniform(self.obstacle_constraints_longer_side[0], self.obstacle_constraints_longer_side[1]))
        v2 = int(random.uniform(self.obstacle_constraints_shorter_side[0], self.obstacle_constraints_shorter_side[1]))
        w = width_favored*v1 + (not width_favored)*v2
        h = width_favored*v1 + (not width_favored)*v2
        if width_favored: 
            w = v1
            h = v2
        else:
            w = v2
            h = v1

        range_of_motion, axis = utils.calculate_allowed_range(w, h)
        lower_bound_x = range_of_motion if axis == "x" else 0
        upper_bound_x = self.world_width -  range_of_motion if axis == "x" else self.world_width
        lower_bound_y = range_of_motion if axis == "y" else 0
        upper_bound_y = self.world_height -  range_of_motion if axis == "y" else self.world_height
        x = int(random.uniform(lower_bound_x+w, upper_bound_x-w))
        y = int(random.uniform(lower_bound_y+h, upper_bound_y-h))
        return w,h,range_of_motion,axis,x,y

    def check_for_collisions(self, x, y, w, h, range_of_motion, axis):
        collides_with_agent = utils.do_collide(x, y, w, h, *self.agent.get_entity_spatial_attributes()[:4])
        
        new_x = x - (axis=="x")*range_of_motion
        new_width = w + (axis=="x")*range_of_motion
        new_y = y - (axis=="y")*range_of_motion
        new_height = h + (axis=="y")*range_of_motion
        collides_with_goal = utils.do_collide(new_x, new_y, new_width, new_height, *self.goal.get_entity_spatial_attributes()[:4])
        
        collides_with_an_obstacle = False
        for obstacle in self.obstacles:
            o_x, o_y, o_w, o_h, o_axis, o_direction = obstacle.get_entity_spatial_attributes()
            collides_with_an_obstacle = collides_with_an_obstacle or utils.do_collide(x, y, w, h, o_x, o_y, o_w, o_h, margin=self.agent_side_length*self.passage_space_factor)
            if collides_with_an_obstacle:
                break

        return collides_with_agent or collides_with_an_obstacle or collides_with_goal



    def get_current_state(self):

        state_list = list(self.agent.get_entity_spatial_attributes())
        state_list.extend(list(self.goal.get_entity_spatial_attributes()))
        for obstacle in self.obstacles:
            state_list.extend(obstacle.get_entity_spatial_attributes())

        state_list.append(self.terminated)

        return state_list

    def _stop_world(self):
        self.agent.set_unit_of_motion(0)
        for obstacle in self.obstacles:
            obstacle.set_unit_of_motion(0)

    ##TODO: better be a dict
    def next_state(self, action=None, uom=None):
        if action is None and uom is None:
            actions, uoms = self.sequence_of_motion
            action = World.ACTIONS_MAP[actions[self.iterator]]
            uom = uoms[self.iterator]
            self.iterator+=1
            self.agent.set_unit_of_motion(uom)
            self.agent.next_state(action)
            for obstacle in self.obstacles:
                obstacle.next_state()
            return None

        

        if self.terminated:
            return self.get_current_state()
        
        self.agent.set_unit_of_motion(uom)
        self.agent.next_state(action)
        for obstacle in self.obstacles:
            obstacle.next_state()

        self._check_for_termination()

        return self.get_current_state()

    def _check_for_out_of_bound(self):
        x, y, w, h, _ = self.agent.get_entity_spatial_attributes()
        return (x <= 0) or ((x + w) >= self.world_width)  or (y <= 0) or ((y + h) >= self.world_height)
        

    def _check_for_termination(self):
        #TODO: CONSIDER AN ENUM FOR TERMINATION STATES IF THEY PROVE USEFUL
        if utils.do_collide(*self.agent.get_entity_spatial_attributes()[:4], *self.goal.get_entity_spatial_attributes()[:4]):
            self.terminated = 2
            return
        
        terminated = False
        for obstacle in self.obstacles:
            o_x, o_y, o_w, o_h, axis, direction = obstacle.get_entity_spatial_attributes()
            # collides_with_an_obstacle = collides_with_an_obstacle or utils.do_collide(x, y, w, h, o_x, o_y, o_w, o_h, margin=self.agent_side_length*self.passage_space_factor)
            terminated = terminated or utils.do_collide(*self.agent.get_entity_spatial_attributes()[:4], o_x, o_y, o_w, o_h)

        terminated = terminated or self._check_for_out_of_bound()

        if terminated:
            self.terminated = 1
    def get_world_initial_properties(self):

        agent_initial_position = (self.agent.x_pos, self.agent.y_pos)
        goal_initial_position = (self.goal.x_pos, self.goal.y_pos)
        
        #TODO: PUT REST OF THE WORLDS PROPERTIES LIKE INITIAL UOM anD AGENT SIDE LENGTH ....
        return {
            "agent_initial_position": agent_initial_position,
            "goal_initial_position": goal_initial_position,
            "obstacles": [
                {
                    "x_pos": o.x_pos,
                    "y_pos": o.y_pos,
                    "width": o.width,
                    "height": o.height,
                    "unit_of_motion": o.unit_of_motion,
                    "range_of_motion": o.range_of_motion,
                    "axis_of_motion": o.axis_of_motion,
                } for o in self.obstacles
            ] 
        }
    # def simulate(self) -> None: #THIS ISN'T HOW SIMULATE WORK
        # self.sequence_of_motion[self.iterator]

        # self.next_state([0, 1], 1)

    def set_sequence_of_motion(self, sequence_of_motion):
        self.iterator = 0
        self.sequence_of_motion = sequence_of_motion

    @classmethod
    def from_prebuilt_world(cls, world_height, world_width, agent_side_length, world_info):
        world = cls.__new__(cls)          # new instance, skips __init__ (which would generate a random world)

        world.world_height = world_height
        world.world_width = world_width
        world.agent_side_length = agent_side_length
        world.unit_of_motion = 1
        world.terminated = False
        world.margin = 50
        x, y = world_info["agent_initial_position"]
        world.agent = Agent(x, y, agent_side_length, agent_side_length, world.unit_of_motion, 0)

        x, y = world_info["goal_initial_position"]
        world.goal = Agent(x, y, agent_side_length, agent_side_length, world.unit_of_motion, 0)

        world.obstacles = [Obstacle(**oci) for oci in world_info["obstacles"]]
        return world

    # def create_obstacles(self):
    #     self.obstacles = []
    #     for i in range(self.n_obstacles):
    #         while True:
    #             w, h, range_of_motion, axis, x, y = self.create_obstacle_spatial_attributes()

    #             if self.check_for_collisions(x, y, w, h, range_of_motion, axis) == False: #No Collisions
    #                 break
    #         print(f"{i+1} obstacles have been created")
    #         obstacle = Obstacle(x, y, w, h, self.unit_of_motion, range_of_motion, axis)#NEEDS FURTHER VALIDATION TO ENSURE IT WONT COLLIDED WITH GOAL WHILE MOVING)
    #         self.obstacles.append(obstacle)
