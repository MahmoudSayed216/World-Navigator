# load model
# instantiate environment
# instantiate agent
# instantiate UI
# update UI via a callback or sth
# TODO: SHOW NN IN ACTION
# TODO: MAKE OBSTACLES OF FIXED SIZE, THUS YOU'LL ONLY HAVE TO PASS THEIR POSITION TO THE NN
# TODO: USE SOME VISUALS
# TODO: Auto Generate environment or Manually creating the environment

from environment import Environment
from agent import Agent
from ui import UI

agent = Agent(20, 20)
environment = Environment(25, 625, 25, 625, agent)

ui = UI(environment, 1300, 700)

ui.main_loop()