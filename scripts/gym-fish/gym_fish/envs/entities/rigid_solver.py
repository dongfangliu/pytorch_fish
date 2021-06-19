
from ..py_util import flare_util
from .underwater_agent import *

import numpy as np

class rigid_solver:
    def __init__(self,rigid_data:flare_util.rigid_data) -> None:
        self._rigid_data = rigid_data
        self._rigid_world =  rigid_data.rigidWorld
        self._agents = [underwater_agent(skeleton_data=sk) for sk in rigid_data.skeletons]
    def get_agent(self,i):
        return self._agents[i]
    def set_commands(self,commands:np.array):
        cmd_offset= 0
        for agent in self._agents:
            agent.set_commands(commands[cmd_offset:cmd_offset+agent.dofs])
            cmd_offset = cmd_offset+agent.dofs
    @property 
    def gravity(self):
        return self._rigid_data.gravity
    @property
    def agent_num(self):
        return len(self._agents)
    @property
    def dt(self):
        return self._rigid_world.getTimeStep()
    @property
    def time(self):
        return self._rigid_world.time