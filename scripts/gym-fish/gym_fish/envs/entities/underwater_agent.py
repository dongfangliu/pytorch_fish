from numpy.core.arrayprint import _void_scalar_repr
from numpy.lib.function_base import select
from ..py_util import flare_util
from ..lib import pyflare as fl
import numpy as np

class fluid_sensor(fl.Marker):
    pass

class buoyancy_control_unit:
    def __init__(self,volume_init:float=1,volume_min:float=0.1,volume_max:float=1.3,control_min:float=0.02,control_max:float=0.1) -> None:
        self.volume_init = volume_init        
        self.volume = volume_init
        self.volume_min = volume_min
        self.volume_max = volume_max
        self.control_min = control_min
        self.control_max = control_max
    def change(self,delta:float)->None:
        volume  =  self.volume-np.clip(delta,self.control_min,self.control_max)
        volume = np.clip(volume,self.volume_min,self.volume_max)
        self.volume = volume
    def reset(self)->None:
        self.volume = self.volume_init
    @property
    def diff_from_init(self)->float:
        return self.volume-self.volume_init

class underwater_agent:
    def __init__(self,skeleton_data:flare_util.skeleton_data) -> None:
        self._dynamics = skeleton_data.dynamics
        self.body_density = skeleton_data.density
    def _setup_bcu(self,volume_min:float,volume_max:float,control_min:float,control_max:float):
        self.bcu = buoyancy_control_unit(1,volume_min,volume_max,control_min,control_max)
        water_den = 1000
        bcu_unit_volume_init = self.mass*(1-water_den/self.body_density)/(water_den)
        self.bcu_amplify_ratio = bcu_unit_volume_init
        # will control in a normalized scheme, while 
    @property
    def buoyancy_force(self)->np.array:
        return self.bcu.diff_from_init*1000*np.array(0,-9.81,0)*self.bcu_amplify_ratio
    @property
    def mass(self):
        return self._dynamics.getMass()
    @property
    def dofs(self):
        pass
    @property
    def position(self):
        pass
    @property
    def linear_vel(self):
        pass
    @property
    def angular_vel(self):
        pass
    @property
    def linear_accel(self):
        pass
    @property
    def angular_accel(self):
        pass
    
    def get_sensors(self):
        
        