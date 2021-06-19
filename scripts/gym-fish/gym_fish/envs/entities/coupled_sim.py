from .fluid_solver import fluid_solver
from .rigid_solver import rigid_solver
import numpy as np
class coupled_sim:
    def __init__(self,fluid_solver:fluid_solver,rigid_solver:rigid_solver) -> None:
        self.rigid_solver = rigid_solver
        self.fluid_solver = fluid_solver
        self.fluid_solver.attach(self.rigid_solver)
    @property
    def dt(self):
        return self.rigid_world.dt
    @property
    def time(self):
        return self.rigid_world.time
    def save(self,save_objects:bool=False,save_fluid:bool=False,suffix:str="0000"):
        self.fluid_solver.save(save_fluid=save_fluid,save_objects=save_objects,suffix=suffix)
    def step(self,commands:np.array):
        self.rigid_solver.set_commands(commands)
        self._simulator.step()