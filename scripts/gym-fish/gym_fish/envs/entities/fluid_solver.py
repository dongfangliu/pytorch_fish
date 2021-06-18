from ..py_util import flare_util
from .rigid_world import rigid_world
class fluid_solver:
    def __init__(self,fluid_parm:flare_util.fluid_param) -> None:
        pass

    def attach(self,_rigid_world:rigid_world)-> None:
        self.rigid_world = rigid_world

    @property
    def dt(self):
        return self.rigid_world.dt
    def step(self):
        pass



