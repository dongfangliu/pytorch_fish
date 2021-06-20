import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import math
from gym_fish.envs.entities.coupled_sim import coupled_sim
from gym_fish.envs.entities.fluid_solver import fluid_solver
from gym_fish.envs.entities.rigid_solver import rigid_solver
import numpy as np
import gym_fish.envs.py_util.flare_util as fl_util
import gym_fish.envs.lib.pyflare as fl
import gym_fish.envs.entities

class coupled_env(gym.Env):
    def __init__(self,fluid_json:str,rigid_json:str,gpuId:int,couple_mode:fl.COUPLE_MODE =fl.COUPLE_MODE.TWO_WAY) -> None:
        super().__init__()
        self.fluid_json =fluid_json
        self.rigid_json = rigid_json
        self.gpuId =  gpuId
        self.couple_mode  = couple_mode
        self.setupDynamics()
    def setupDynamcis(self):
        fluid_param =fl_util.fluid_param()
        fluid_param.from_json(self.fluid_json)
        rigids_data = fl_util.rigid_data(gpuId=self.gpuId)
        rigids_data.from_json(self.rigid_json)

        rigid_solv = rigid_solver(rigids_data)
        fluid_solv = fluid_solver(fluid_param = fluid_param,gpuId=self.gpuId,couple_mode=self.couple_mode)
        self.simulator = coupled_sim(fluid_solv,rigid_solv)
    
    def step(self, action):
        self.simulator.step(action)





