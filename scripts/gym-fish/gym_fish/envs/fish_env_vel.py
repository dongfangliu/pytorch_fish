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
class FishEnvVel(gym.Env):
    metadata = {}
    def __init__(self, gpuId:int = 0,
                 control_dt=0.2,
                 wv = 1.0,
                 wd= 0.5,
                 wa=0.00001,
                 live_penality=-0.1,
                 couple_mode= fl.COUPLE_MODE.TWO_WAY,
                 theta = np.array([-45,45]),
                 vel_theta = np.array([-45.45]),
                 random_vel = np.array([0,0.3]),
                 action_max = 5,
                 max_time = 10,
                 done_dist=0.1,
                 use_com=True,
                 rigid_json: str='./py_data/jsons/rigids_2_30.json',
                 fluid_json: str='./py_data/jsons/fluid_param_0.5.json',
                ):
        # store input args
        super().__init__()
        self.gpuId = gpuId
        self.rigid_json = rigid_json
        self.fluid_json = fluid_json
        
        self.random_vel = random_vel
        self.couple_mode = couple_mode
        self.action_max = action_max
        self.max_time = max_time
        self.theta = theta/180.0*math.pi
        self.vel_theta = vel_theta/180.0*math.pi
        self.control_dt = control_dt
        
        self.wv = wv
        self.wd = wd
        self.wa = wa
        self.live_penality = live_penality
        self.done_dist = done_dist
        self.control_dt=control_dt
        self.training = True
        self.use_com= use_com
        
        
        self.setupDynamcis()
        
        self.seed()
        self.set_datapath('./data/vis_data/')
        ## set action and state space
        self.dofs = []
        for skeleton in self.rigid_data.skeletons:
            self.dofs.append(skeleton.dynamics.getNumDofs())
#             self.renderer.addDynamics(skeleton.dynamics)
            
        self.action_dim = sum(self.dofs)
        
        
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
        
        

#     def set_path(self,path:flare_util.path_data):
#         assert type(path)==flare_util.path_data
#         self.path_data=path
        
    def stepOnce(self,raw_action,save_fluid=False, save_objects=True):
        dynamics = self.rigid_data.skeletons[0].dynamics
        dt = self.rigid_data.rigidWorld.getTimestep()
#         dynamics.setAngleCommands(raw_action,dt)
        dynamics.setCommands(raw_action)
        self.simulator.step(self.couple_mode)
        obs = self._get_obs()
        self.trajectory_points.append(self.body_xyz)
#         reward,info = self._get_reward()
        done = self._get_done() 
        if (not np.isfinite(obs).all()):
            print("NAN OBS")
            obs = self.last_obs
            done=True
        self.save_data_framRate(save_fluid,save_objects)
        self.last_obs = obs
#         return obs,reward,done,info
        return obs,None,done,None
    
    def step(self, action,save_fluid=False, save_objects=False,test_mode=False):
        self.last_action = action
        act= self._unnormalize_action(action)
        dt = self.rigid_data.rigidWorld.getTimestep()
        dynamics = self.rigid_data.skeletons[0].dynamics
        t = 0
        while t< self.control_dt:
            ctrl = act
            obs,_,done,_ =self.stepOnce(ctrl,save_fluid,save_objects)
            t = t+dt
            if done:
                if not self.training:
                    continue;
                else:
                    break
        reward,info = self._get_reward(obs = obs,action=action)
        return obs, reward, done,info
#         o,r,d,i = self.stepOnce(action,save_fluid, save_objects)
#         reward,info = self._get_reward()
#         return o,reward,d,info

    def stepSave(self,action,save_fluid=False, save_objects=True,test_mode=False):
        o,r,d,i = self.step(action,save_fluid, save_objects,test_mode=test_mode)
        return o,r,d,i
    

    def _get_done(self):
        done = False 
        done = done or self.rigid_data.rigidWorld.time>self.max_time 
#         done = done or np.linalg.norm(self.dist_to_path)>0.8
        return  done 


    
    def _get_reward(self,obs=None, action=None):
        dynamics = self.rigid_data.skeletons[0].dynamics
        vel = dynamics.getCOMLinearVelocity()
        vel_diff_norm  =np.exp(- np.linalg.norm(vel-self.goal_vel)/0.2)
        vel_reward = self.wv* vel_diff_norm
        
        action_reward =-np.sum(np.abs(action)**0.5)*self.wa
        cur_dir = vel/(np.linalg.norm(vel)+1e-6)
#         cur_dir = self.rigid_data.skeletons[0].dynamics.getBaseLinkFwd()
    
        
        ori_reward = self.wd* (np.dot(cur_dir,self.goal_dir)+1)/2
        
        
        total_reward = self.live_penality+action_reward+ori_reward+vel_reward
        
        info = {'live_penality':self.live_penality,"vel_reward":vel_reward,"action_reward":action_reward,'ori_reward':ori_reward}
        return min(max(-5,total_reward),5),info
    def _get_action_space(self):
#         low = np.array([-self.action_max,-self.action_max,-self.action_max,-self.action_max])
#         high=np.array([self.action_max,self.action_max,self.action_max,self.action_max])
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
            vel  =  dynamics.getCOMLinearVelocity()
        else:
            self.body_xyz =  dynamics.getBaseLink().getPosition()  
            vel  =  dynamics.getBaseLink().getLinearVelocity()
        
        
        # update local matrix
        x_axis = dynamics.getBaseLinkFwd()
        y_axis = dynamics.getBaseLinkUp()
        z_axis = dynamics.getBaseLinkRight()
        self.world_to_local = np.linalg.inv(np.array([x_axis,y_axis,z_axis]).transpose())
        self.rpy = np.arccos(np.array([x_axis[0],y_axis[1],z_axis[2]]))
        
                             
        self.angle_to_target = np.arccos(np.dot(x_axis, self.goal_dir ))
        if np.dot(self.goal_dir, dynamics.getBaseLinkRight())<0:
            self.angle_to_target = -self.angle_to_target
        #in local coordinate
        vel_local = np.dot(self.world_to_local,np.transpose(vel))
        dvel_local = np.dot(self.world_to_local,np.transpose(vel-self.goal_vel))
        
        
            
        joint_pos = dynamics.getPositions(includeBase=False)
        joint_vel = dynamics.getVelocities(includeBase=False)
        obs = np.concatenate(
            (
                np.array([self.angle_to_target/math.pi]),
                dvel_local,
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
    def set_task(self,theta,vel):
        self.goal_dir = np.array([math.cos(theta),0,math.sin(theta)])
        self.goal_pos = self.init_pos+2*self.goal_dir
        self.goal_vel = self.goal_dir*vel
        
    def _reset_task(self):
        theta = self.np_random.uniform(self.theta[0],self.theta[1])
        vel = self.np_random.uniform(self.random_vel[0],self.random_vel[1],size=1)[0]
        self.set_task(theta,vel)
        
        
    def reset(self):
        self.setupDynamcis()
        self._reset_robot()
        self._reset_task()
        
        self.trajectory_points=[]
        
        self.last_action = np.ones(self.action_dim)*0
        self.last_obs = self._get_obs()
        
        
        ref_line = fl.debugLine()
        ref_line.vertices = [
            self.body_xyz*(1.0-t)+self.goal_pos*t for t in np.arange(0.0,1.0,1.0/100)
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
#         if path_points!=None:
#             ax.scatter3D(xs=[x.data[0] for x in path_points], zs=[x.data[1] for x in path_points], ys=[x.data[2] for x in path_points],c='g')
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
