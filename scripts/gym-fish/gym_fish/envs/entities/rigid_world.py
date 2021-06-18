
from ..py_util import flare_util
from .underwater_agent import *

import numpy as np

class rigid_world:
    def __init__(self,rigid_data:flare_util.rigid_data) -> None:
        self._rigid_data = rigid_data
    @property 
    def gravity(self):
        return self._rigid_data.gravity
    @property
    def agent_num(self):
        return len(self._rigid_data.skeletons)
    @property
    def dt(self):
        return self._rigid_data.rigidWorld.getTimeStep()
    