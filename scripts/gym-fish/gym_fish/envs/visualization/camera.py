from typing import Tuple
import numpy as np
from pyrr import Matrix44, Quaternion, Vector3, vector
class camera:
    def __init__(self,
    z_near=0.1,
    z_far=1000,fov=60,
    center = np.array([0,0,0]),
    up = np.array([0,1,0]),
    target = np.array([0,0,1]),
    window_size:Tuple[float]=(1280,720)
    ) -> None:
        self._z_near = z_near
        self._z_far = z_far
        self._aspect_ratio = window_size[0]/window_size[1]
        self._fov = fov
        self.window_size = window_size
        self._center = center
        self._up = up
        self.set_target(target=target)
        self.build_projection()
        self.build_look_at()
        
    def build_look_at(self):
        self._target = (self._center + self._fwd)
        self.mat_lookat = Matrix44.look_at(
            self._center,
            self._target,
            self._up)
    def set_target(self,target:np.array = np.array([0,0,1]))->None:
        self.fwd = (target-self.center)/np.linalg.norm(target-self.center)
        self.right = np.cross(self.up,self.fwd)
    def build_projection(self):
        self.mat_projection = Matrix44.perspective_projection(
            self._fov,
            self._aspect_ratio,
            self._z_near,
            self._z_far)
    @property
    def mvp(self):
        self.build_look_at()
        self.build_projection()
        return self.mat_projection * self.mat_lookat

