import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import math
from collections import OrderedDict,deque
import numpy as np
from numpy import *
from .py_util import flare_util
from .lib import pyflare as fl
from pathlib import Path
import copy

from .py_util.np_util import generate_traj

from pathlib import *
                
import matplotlib.pyplot as plt      
from mpl_toolkits import mplot3d
import traceback

from gym.wrappers import TimeLimit
class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.
    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps=5000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high= np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.
        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class FishEnv(gym.Env):
    metadata = {}
    def __init__(self, gpuId:int = 0,
                 control_dt=0.2,
                 radius=2.0,
                 wp= np.array([0.0,1.0]),
                 wv = 0.0,
                 wa=0.00001,
                 live_penality=-0.1,
                 couple_mode= fl.COUPLE_MODE.TWO_WAY,
                 step_budget_range=np.array([0,1]),
                 theta = np.array([-45,45]),
                 random_vel = np.array([0,0.3]),
                 action_max = 2,
                 max_time = 10,
                 done_dist=0.3,
                 use_com=True,
                 rigid_json: str='./py_data/jsons/rigids_2_30.json',
                 fluid_json: str='./py_data/jsons/fluid_param_0.5.json',
                ):
        # store input args
        super().__init__()
        self.gpuId = gpuId
        self.rigid_json = rigid_json
        self.fluid_json = fluid_json
        self.radius = radius
        self.random_vel = random_vel
        self.couple_mode = couple_mode
        self.action_max = action_max
        self.max_time = max_time
        
        self.theta = theta/180.0*math.pi
        self.step_budget_range =step_budget_range
        
        self.wv = wv
        self.wp = wp
        self.wa = wa
        self.live_penality = live_penality
        self.done_dist = done_dist
        self.control_dt=control_dt
        self.training = True
        self.use_com= use_com
        
        
        self.setupDynamcis()
        
        self.seed()
        self.set_datapath('./data/vis_data/')
        
        
        _obs = self.reset()
        self.action_space = self._get_action_space() 
        self.observation_space = convert_observation_to_space(_obs)
        ## Check saving directory avaliable

    def set_datapath(self,folder_path):
        self.simulator.mainDataFolderPath = folder_path
        self.dataPath = {}
        self.dataPath["fluid"] = str(Path(self.simulator.mainDataFolderPath + self.simulator.fluidFolderName + '/').resolve())
        self.dataPath["objects"] = str(Path(self.simulator.mainDataFolderPath + self.simulator.objectsFolderName + '/').resolve())
        self.dataPath["trajectory"] = str(Path(self.simulator.mainDataFolderPath + 'Trajectory/').resolve())
        
        if not os.path.exists(self.simulator.mainDataFolderPath):
            os.makedirs(self.simulator.mainDataFolderPath)
        for p in self.dataPath.values():
            if not os.path.exists(p):
                os.makedirs(p)
                
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    
    def setupDynamcis(self):
        #fluid
        fluid_param = flare_util.fluid_param()
        fluid_param.from_json(self.fluid_json)
        # Rigid
        rigids = flare_util.rigid_data(gpuId=self.gpuId)
        rigids.from_json(self.rigid_json)
        self.fluid_param = fluid_param
        # generate data from input args
#         self.path_data = path
        self.rigid_data = rigids
        self.simulator = fl.make_simulator(self.fluid_param.data,self.gpuId)
        self.simulator.attachWorld(self.rigid_data.rigidWorld)
        self.simulator.commitInit()
        
        self.dt  = self.rigid_data.rigidWorld.getTimestep()
#         self.simulator.log()
        
        
    def close(self):
        del self.simulator
#         del self.path_data
        del self.rigid_data
        del self.fluid_param
#         del self.renderer
        
        
        
    def stepOnce(self,raw_action,save_fluid=False, save_objects=True):
        dynamics = self.rigid_data.skeletons[0].dynamics
        dt = self.rigid_data.rigidWorld.getTimestep()
        dynamics.setCommands(raw_action)
        self.simulator.step(self.couple_mode)
        obs = self._get_obs()
        self.trajectory_points.append(self.body_xyz)
        done = self._get_done() 
        if (not np.isfinite(obs).all()):
            print("NAN OBS")
            obs = self.last_obs
            done=True
        self.save_data_framRate(save_fluid,save_objects)
        self.last_obs = obs
        return obs,None,done,None
    
    def step(self, action,save_fluid=False, save_objects=False,test_mode=False):
        dt = self.rigid_data.rigidWorld.getTimestep()
        dynamics = self.rigid_data.skeletons[0].dynamics
        t = 0
        action = self._update_budget_action(action)
        self.last_action = action
        act= self._unnormalize_action(action)
        while t< self.control_dt:
            obs,_,done,_ =self.stepOnce(act,save_fluid,save_objects)
            t = t+dt
            if done:
                if not self.training:
                    continue;
                else:
                    break
        reward,info = self._get_reward(self.control_dt,t,obs = obs,action=action)
        return obs, reward, done,info

    def stepSave(self,action,save_fluid=False, save_objects=True,test_mode=False):
        o,r,d,i = self.step(action,save_fluid, save_objects,test_mode=test_mode)
        return o,r,d,i
    

    def _get_done(self):
        done = False 
        done = done or self.rigid_data.rigidWorld.time>self.max_time 
        done = done or np.linalg.norm(self.body_xyz-self.goal_pos)<self.done_dist
        return  done 

    def _update_budget_action(self,action):
        # return feasible action in budget
        attempt_cost = np.sum(np.abs(action))
        feasible_cost = np.clip(attempt_cost,0,self.cur_budget)
        action = action*feasible_cost/(attempt_cost+1e-6)
        self.action_cost = feasible_cost
        self.cur_budget=max(0,self.cur_budget-feasible_cost)
        return action
    
    def calc__dist_potential(self):
        return -self.walk_target_dist / 0.05*(0.2/(self.control_dt))
    

    
    def _get_reward(self,t_standard,t, obs=None, action=None):
        dist_potential_old = self.dist_potential
        self.dist_potential = self.calc__dist_potential()
        dist_reward = self.wp[0]*np.exp(-3* (self.walk_target_dist**2))+self.wp[1]*float(self.dist_potential - dist_potential_old)/t*t_standard
        action_reward =-np.sum(np.abs(action)**0.5)*self.wa
        vel_reward = self.wv*np.exp(-5* np.abs(np.linalg.norm(self.vel)-self.target_vel))
        
        total_reward = dist_reward+action_reward+vel_reward+self.live_penality
        
        info = {'live_penality':self.live_penality,'dist_reward':dist_reward,"action_reward":action_reward,'vel_reward':vel_reward}
        return min(max(-5,total_reward),5),info
    
    def _get_action_space(self):
        low = -np.ones(self.action_dim)*self.action_max
        high= -low
        self.action_space_mean = (low+high)/2
        self.action_space_std = (high-low)/2
        return spaces.Box(low = -1,high=1,shape=low.shape)
    
    def _normalize_action(self,action):
        return np.clip((action-self.action_space_mean)/self.action_space_std,-1,1)
    def _unnormalize_action(self,action):
        action = np.clip(action,-1,1)
        return action*self.action_space_std+self.action_space_mean
    
        
    def _get_obs(self):
        # obs
        dynamics = self.rigid_data.skeletons[0].dynamics
        if self.use_com==True:
            self.body_xyz =  dynamics.getCOM()
            self.vel  =  dynamics.getCOMLinearVelocity()
        else:
            self.body_xyz =  dynamics.getBaseLink().getPosition()  
            self.vel  =  dynamics.getBaseLink().getLinearVelocity()
        
        
        # update local matrix
        x_axis = dynamics.getBaseLinkFwd()
        y_axis = dynamics.getBaseLinkUp()
        z_axis = dynamics.getBaseLinkRight()
        self.world_to_local = np.linalg.inv(np.array([x_axis,y_axis,z_axis]).transpose())
        self.rpy = np.arccos(np.array([x_axis[0],y_axis[1],z_axis[2]]))
        
                             
        self.walk_target_dist = np.linalg.norm(self.body_xyz-self.goal_pos)
        self.angle_to_target = np.arccos(np.dot(x_axis, (self.goal_pos-self.body_xyz)/self.walk_target_dist ))
        if np.dot((self.goal_pos-self.body_xyz)/self.walk_target_dist, dynamics.getBaseLinkRight())<0:
            self.angle_to_target = -self.angle_to_target
        #in local coordinate
        dp_local = np.dot(self.world_to_local,np.transpose(self.goal_pos-self.body_xyz))
        vel_local = np.dot(self.world_to_local,np.transpose(self.vel))
        
            
        
        joint_pos = dynamics.getPositions(includeBase=False)
        joint_vel = dynamics.getVelocities(includeBase=False)
        
        task_config = np.array([self.step_budget,self.target_vel,self.dp_init[0],self.dp_init[1],self.dp_init[2]])
        
        obs = np.concatenate(
            (
                task_config,
                dp_local,
                vel_local,
                
#                 self.last_action,
                
                joint_pos/0.52,
                joint_vel/10,
                
        ),axis=0)
        return obs
    

        
        
    def _reset_robot(self):
        skeleton_dynamics = self.rigid_data.skeletons[0].dynamics
#         angle =  self.np_random.uniform(self.vel_theta[0],self.vel_theta[1])
#         vel =  np.array([math.cos(angle),0,math.sin(angle)])*self.np_random.uniform(self.random_vel[0],self.random_vel[1],size=1)
        
#         skeleton_dynamics.getJoint("head").setVelocity(0,vel[0])
#         skeleton_dynamics.getJoint("head").setVelocity(1,vel[1])
# #         skeleton_dynamics.getJoint("head").setPosition(2,self.np_random.uniform(-0.52,0.52,size=1))
#         joint_list =['spine','spine01','spine02','spine03']
#         for jnt_name in joint_list:
#             skeleton_dynamics.getJoint(jnt_name).setPosition(0,self.np_random.uniform(-0.52,0.52,size=1))
#             skeleton_dynamics.getJoint(jnt_name).setVelocity(0,self.np_random.uniform(-10,10,size=1))
#         skeleton_dynamics.update()
        self.body_xyz = skeleton_dynamics.getBaseLink().getPosition()
        self.init_pos = self.body_xyz
    def set_task(self,goal_pos,target_vel,step_budget):
        dynamics = self.rigid_data.skeletons[0].dynamics
        x_axis = dynamics.getBaseLinkFwd()
        y_axis = dynamics.getBaseLinkUp()
        z_axis = dynamics.getBaseLinkRight()
        self.goal_pos = goal_pos
        self.world_to_local = np.linalg.inv(np.array([x_axis,y_axis,z_axis]).transpose())
        self.dp_init = np.dot(self.world_to_local,np.transpose(self.goal_pos-self.init_pos))
        self.target_vel = target_vel
        self.step_budget = step_budget
        self.init_budget = step_budget*self.max_time/self.control_dt*self.action_dim
        self.cur_budget = step_budget*self.max_time/self.control_dt*self.action_dim
        print('Set task with BUDGET {} ,VEL {},GOAL {}'.format(self.step_budget,self.target_vel,self.dp_init))
        
    def _reset_task(self):
        theta = self.np_random.uniform(self.theta[0],self.theta[1])

        goal_dir = np.array([math.cos(theta),0,math.sin(theta)])
        goal_pos = self.init_pos+self.radius*goal_dir
    
        target_vel = self.np_random.uniform(self.random_vel[0],self.random_vel[1])
        budget = self.np_random.uniform(self.step_budget_range[0],self.step_budget_range[1])
        
        self.set_task(goal_pos,target_vel,budget)
        
        
    def reset(self):
        self.setupDynamcis()
        
        ## set action and state space
        self.dofs = []
        for skeleton in self.rigid_data.skeletons:
            self.dofs.append(skeleton.dynamics.getNumDofs())
#             self.renderer.addDynamics(skeleton.dynamics)
            
        self.action_dim = sum(self.dofs)
        
        self._reset_robot()
        self._reset_task()
        
        self.trajectory_points=[]
        
        self.last_action = np.ones(self.action_dim)*0
        self.last_obs = self._get_obs()
        self.dist_potential = self.calc__dist_potential()
        
        
        ref_line = fl.debugLine()
        ref_line.vertices = [
            self.init_pos*(1.0-t)+self.goal_pos*t for t in np.arange(0.0,1.0,1.0/100)
        ]
        fl.VTKWriter.writeLines([ref_line], self.dataPath["trajectory"]+"/trajectory_ideal.vtk")
        return self.last_obs
    
    
    
    def plot3d(self,title=None,fig_name=None,elev=45,azim=45):
        path_points =np.array( [
            self.init_pos*(1.0-t)+self.goal_pos*t for t in np.arange(0.0,1.0,1.0/100)
        
        ])
        trajectory_points = self.trajectory_points
        ax=plt.figure().add_subplot(111, projection = '3d')
        X = path_points[:,0]
        Y = path_points[:,1]
        Z = path_points[:,2]
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_z - max_range, mid_z + max_range)
        ax.set_zlim(mid_y - max_range, mid_y + max_range)
        ax.scatter3D(xs=X, zs=Y, ys=Z,c='g')
        if trajectory_points!=None:
            ax.scatter3D(xs=[x[0] for x in trajectory_points],
                zs=[x[1] for x in trajectory_points],
                ys=[x[2] for x in trajectory_points],
                c=[[0,0,i/len(trajectory_points)] for i in range(len(trajectory_points))])
        ax.view_init(elev=elev,azim=azim)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        if title!=None:
            ax.set_title(title)
        if fig_name!=None:
            plt.savefig(fig_name)
        plt.show()
    def render(self, mode='human'):
        if self.renderer!=None:
            self.renderer.render()
    def save_data(self, save_fluid=False, save_objects=True,frame_num = 0):
        ref_line = fl.debugLine()
        ref_line.vertices = self.trajectory_points
        fl.VTKWriter.writeLines([ref_line], self.dataPath["trajectory"]+"/trajectory_actual%05d.vtk"% frame_num)
        if save_fluid:
            fluid_name = "fluid%05d" % frame_num
            self.simulator.saveFluidData(fluid_name)
        if save_objects:
            objects_name = "object%05d" % frame_num
            self.simulator.saveObjectsData(objects_name)
        
    def save_data_framRate(self, save_fluid=False, save_objects=True,framRate = 30):
        if self.simulator.getIterNum()%self.simulator.getIterPerSave(framRate)!=0:
            return
        frame_num=(self.simulator.getIterNum() / self.simulator.getIterPerSave(framRate))
        self.save_data(save_fluid, save_objects,frame_num)
