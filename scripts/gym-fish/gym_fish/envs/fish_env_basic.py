from sys import path
from typing import Any, Dict, Tuple
from .coupled_env import coupled_env
import gym_fish.envs.lib.pyflare as fl
import gym_fish.envs.py_util.np_util as np_util 
import numpy as np
import os
import math
import pathlib

def get_full_path(asset_path):
    if asset_path.startswith("/"):
        full_path = asset_path
    else:
        full_path = os.path.join(os.path.dirname(__file__), asset_path)
    if not os.path.exists(full_path):
        raise IOError("File %s does not exist" % full_path)
    return full_path
class FishEnvBasic(coupled_env):
    def __init__(self, 
                control_dt=0.2,
                wc = np.array([0.0,1.0]),
                wp= np.array([0.0,1.0]),
                wa=0.5,
                action_max = 5,
                max_time = 10,
                done_dist=0.1,
                radius = 1,
                theta = np.array([-45,45]),
                dist_distri_param =np.array([0,0.5]),
                data_folder = "./data/vis_data/",
                fluid_json: str='../assets/fluid/fluid_param_0.5.json',
                rigid_json: str='../assets/rigid/rigids_4_30_new.json',
                gpuId: int=0,
                couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY) -> None:
        fluid_json = get_full_path(fluid_json)
        rigid_json = get_full_path(rigid_json)
        self.wc = wc
        self.wp = wp
        self.wa = wa
        self.done_dist = done_dist
        self.theta = theta/180.0*math.pi
        self.dist_distri_param = dist_distri_param
        self.control_dt=control_dt
        self.action_max = action_max
        self.max_time = max_time
        self.radius = radius
        self.training = True
        # use parent's init function to init default env data, like action space and observation space, also init dynamics
        super().__init__(fluid_json, rigid_json, gpuId, couple_mode=couple_mode)
        self.simulator.fluid_solver.set_savefolder(pathlib.Path(data_folder).resolve())


    def _step(self, action) -> None:
        t = 0
        while t<self.control_dt:
            self.simulator.iter(action)
            t = t+self.simulator.dt
            if(self._get_done() and not self.training):
                break
    def _get_reward(self, cur_obs, cur_action) :
        dist_potential_old = self.dist_potential
        self.dist_potential = self.calc__dist_potential()
        dist_reward = self.wp[0]*np.exp(-3* (self.walk_target_dist**2))+self.wp[1]*float(self.dist_potential - dist_potential_old)
        
        close_potential_old = self.close_potential
        self.close_potential = self.calc__close_potential()
        close_reward = self.wc[0]*np.exp(-5* self.dist_to_path)+self.wc[1]*float(self.close_potential - close_potential_old)
        
        action_reward = -np.sum(np.abs(cur_action)**0.5)*self.wa
        
        total_reward = dist_reward+close_reward+action_reward
        
        info = {'dist_reward':dist_reward,"action_reward":action_reward,'close_reward':close_reward}
        return min(max(-5,total_reward),5),info

    def _get_done(self) -> bool:
        done = False 
        done = done or self.simulator.time>self.max_time 
        done = done or np.linalg.norm(self.body_xyz-self.goal_pos)<self.done_dist
        done = done or np.linalg.norm(self.dist_to_path)>0.8
        done = done or (not np.isfinite(self._get_obs()).all())
        return  done 

    def _get_obs(self) -> np.array:
        agent = self.simulator.rigid_solver.get_agent(0)
        proj_pt_local = np.dot(self.world_to_local,np.transpose(self.proj_pt_world-self.body_xyz))
        obs = np.concatenate(
            (
                np.array([self.angle_to_target]),
                self.dp_local,
                proj_pt_local,
                self.vel_local,
                agent.positions/0.52,
                agent.velocities/10,
        ),axis=0)
        return obs

    def _update_state(self):
        agent = self.simulator.rigid_solver.get_agent(0)
        self.body_xyz =  agent.com
        vel  =  agent.linear_vel
        # update local matrix
        x_axis = agent.fwd_axis
        y_axis = agent.up_axis
        z_axis = agent.right_axis
        self.world_to_local = np.linalg.inv(np.array([x_axis,y_axis,z_axis]).transpose())
        self.rpy = np.arccos(np.array([x_axis[0],y_axis[1],z_axis[2]]))
        self.walk_target_dist = np.linalg.norm(self.body_xyz-self.goal_pos)
        self.angle_to_target = np.arccos(np.dot(x_axis, (self.goal_pos-self.body_xyz)/self.walk_target_dist ))
        if np.dot((self.goal_pos-self.body_xyz)/self.walk_target_dist,agent.right_axis)<0:
            self.angle_to_target = -self.angle_to_target
        #in local coordinate
        self.dp_local = np.dot(self.world_to_local,np.transpose(self.goal_pos-self.body_xyz))
        self.vel_local = np.dot(self.world_to_local,np.transpose(vel))
        
        rela_vec_to_goal = self.goal_pos-self.body_xyz
        if self.training:
            self.proj_pt_world = self.goal_pos-self.path_dir*np.dot(rela_vec_to_goal,self.path_dir)
        self.dist_to_path = np.linalg.norm(self.proj_pt_world)


    def calc__dist_potential(self):
        return -self.walk_target_dist /self.control_dt/ 4
    def calc__close_potential(self):
        return -self.dist_to_path /self.control_dt/4

    def set_task(self,theta,dist):
        agent = self.simulator.rigid_solver.get_agent(0)
        self.init_pos = agent.com
        goal_dir = np.array([math.cos(theta),0,math.sin(theta)])
        self.goal_pos = self.init_pos+self.radius*goal_dir
        has_sol,start_pts = np_util.generate_traj(self.init_pos,self.goal_pos,dist,visualize=False)
        self.path_start = start_pts[np.random.choice(start_pts.shape[0]),:]
        self.path_start =np.array([self.path_start[0],self.init_pos[1],self.path_start[1]])
        self.path_dir = self.goal_pos-self.path_start
        self.path_dir = self.path_dir/np.linalg.norm(self.path_dir)
        self.path_start = self.goal_pos-self.path_dir*self.radius
    
    def _reset_task(self):

        theta = self.np_random.uniform(self.theta[0],self.theta[1])
        dist = self.np_random.uniform(self.dist_distri_param[0],self.dist_distri_param[1],size=1)[0]
#         dist = self.np_random.normal(self.dist_distri_param[0],self.dist_distri_param[1],size=1)[0]
        dist =min(max(0.01,dist),1.0)
        self.set_task(theta,dist)


    def reset(self) -> Any:
        self.resetDynamics()
        self._reset_task()
        self._update_state()
        self.trajectory_points=[]
        self.dist_potential = self.calc__dist_potential()
        self.close_potential = self.calc__close_potential()
        
        return self._get_obs()
    