import gym
from gym import spaces, utils
from gym.utils import seeding
from collections import OrderedDict
import numpy as np
import os
from numpy import *
from .py_util import flare_util
from .lib import pyflare as fl

from pathlib import *
                
import matplotlib.pyplot as plt

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
    def __init__(self, gpuId:int = 0,frame_skip=200,
                 wr = 0.0,wp=1.0,live_penality=0.0,wa=0.1,
                 couple_mode= fl.COUPLE_MODE.TWO_WAY,
                 theta = 0,
#                  goal_pos = np.array([3,2,3]),
#                  goal_dir = np.array([1,0,0]),
                 path_json: str='./py_data/jsons/paths/line.json', 
                 rigid_json: str='./py_data/jsons/rigids_2_30.json',
                 fluid_json: str='./py_data/jsons/fluid_param_0.5.json',
                ):
        # store input args
        super().__init__()
        self.gpuId = gpuId
        self.path_json = path_json
        self.rigid_json = rigid_json
        self.fluid_json = fluid_json
        
        self.couple_mode = couple_mode
        
        self.theta = theta/180.0*math.pi
        
        self.wr = wr
        self.wp = wp
        self.wa = wa
        self.live_penality = live_penality
#         self.goal_pos = goal_pos
        self.frame_skip=frame_skip
#         self.goal_dir = goal_dir
        
        
        self.setupDynamcis()
        
        self.seed()
        ## setup renderer
#         self.renderer = fl.make_renderer(1024,768)
#         self.renderer.addTrajectory(self.path_data.trajectory)
#         self.renderer.setCamera([2,5.5,4],[2,2,3],[0,1,0])
        
        _obs = self.reset()
        ## set action and state space
        self.dofs = []
        for skeleton in self.rigid_data.skeletons:
            self.dofs.append(skeleton.dynamics.getNumDofs())
#             self.renderer.addDynamics(skeleton.dynamics)
            
        self.action_dim = sum(self.dofs)
        self.action_space = self._get_action_space() 
        self.observation_space = convert_observation_to_space(_obs)
        ## Check saving directory avaliable
        self.dataPath = {}
        self.dataPath["fluid"] = str(Path(self.simulator.mainDataFolderPath + self.simulator.fluidFolderName + '/').resolve())
        self.dataPath["objects"] = str(Path(self.simulator.mainDataFolderPath + self.simulator.objectsFolderName + '/').resolve())
        self.dataPath["trajectory"] = str(Path(self.simulator.mainDataFolderPath + 'Trajectory/').resolve())
        
        if not os.path.exists(self.simulator.mainDataFolderPath):
            os.makedirs(self.simulator.mainDataFolderPath)
        for p in self.dataPath.values():
            if not os.path.exists(p):
                os.makedirs(p)
        # save Trajectory
        fl.VTKWriter.writeTrajectory(self.path_data.trajectory, self.dataPath['trajectory'] + '/trajectory.vtk')
        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    
    def setupDynamcis(self):
        #path
        path = flare_util.path_data()
        path.from_json(self.path_json)
        #fluid
        fluid_param = flare_util.fluid_param()
        fluid_param.from_json(self.fluid_json)
        # Rigid
        rigids = flare_util.rigid_data(gpuId=self.gpuId)
        rigids.from_json(self.rigid_json)
        print(self.gpuId)
        self.fluid_param = fluid_param
        # generate data from input args
        self.path_data = path
        self.rigid_data = rigids
        self.simulator = fl.make_simulator(self.fluid_param.data,self.gpuId)
        self.simulator.attachWorld(self.rigid_data.rigidWorld)
        self.simulator.commitInit()
#         self.simulator.log()
        
        
    def close(self):
        del self.simulator
        del self.path_data
        del self.rigid_data
        del self.fluid_param
#         del self.renderer
        
        

    def set_path(self, path: flare_util.path_data):
        assert type(path) == flare_util.path_data
        self.path_data=path
        
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
            done=True
        self.save_data_framRate(save_fluid,save_objects)
#         return obs,reward,done,info
        return obs,None,done,None
    
    def step(self, action,save_fluid=False, save_objects=False):
#         act= self._unnormalize_action(action)
#         magnitude_0 = act[0]
#         period_0= act[1]
#         magnitude_1 = act[2]
#         period_1= act[3]
        
#         dt = self.rigid_data.rigidWorld.getTimestep()
#         dynamics = self.rigid_data.skeletons[0].dynamics
#         t = 0
#         while t< (period_0+period_1)*0.5:
#             if t<0.5*period_0:
#                 targetAngle = magnitude_0*math.sin(6.28*(1.0/period_0)*t )
#             else:
#                 targetAngle = -magnitude_1*math.sin(6.28*(1.0/period_1)*(t-0.5*period_0) )
#             ctrl = (targetAngle-dynamics.getJoint("spine02").getPositions()[0])/dt
#             obs,_,done,_ =self.stepOnce([ctrl],save_fluid,save_objects)
#             if done:
#                 break
#             t = t+dt
#         reward,info = self._get_reward()
#         return obs, reward, done,info

        act= self._unnormalize_action(action)
        dt = self.rigid_data.rigidWorld.getTimestep()
        dynamics = self.rigid_data.skeletons[0].dynamics
        t = 0
        while t< dt*self.frame_skip:
            ctrl = act
            obs,_,done,_ =self.stepOnce(ctrl,save_fluid,save_objects)
            if done:
                break
            t = t+dt
        reward,info = self._get_reward()
        return obs, reward, done,info
#         o,r,d,i = self.stepOnce(action,save_fluid, save_objects)
#         reward,info = self._get_reward()
#         return o,reward,d,info

    def stepSave(self,action,save_fluid=False, save_objects=True):
        o,r,d,i = self.step(action,save_fluid, save_objects)
        return o,r,d,i
    

    def _get_done(self):
        return self._alive < 0 or self.rigid_data.rigidWorld.time>10

    
    
    def calc__dist_potential(self):
        return -self.walk_target_dist / 0.05
    def calc__dir_potential(self):
        return -self.angle_to_dir /(10.0 /180.0*3.14)
    

    
    def _get_reward(self, obs=None, action=None):
        
        dist_potential_old = self.dist_potential
        self.dist_potential = self.calc__dist_potential()
        dist_reward = float(self.dist_potential - dist_potential_old)
        
        dir_potential_old = self.dir_potential
        self.dir_potential = self.calc__dir_potential()
        
#         chase_dir_reward = float(self.dir_potential - dir_potential_old)
        chase_dir_reward = np.dot(self.rigid_data.skeletons[0].dynamics.getBaseLinkFwd(),self.goal_dir)
    
        angle_reward =-np.linalg.norm(self.rigid_data.skeletons[0].dynamics.getBaseLink().getAngularVelocity())/5
            
        
        total_reward = self.wr*chase_dir_reward+self.wp*dist_reward+self.live_penality+self.wa*angle_reward+self._alive
        info = {'chase_dir_reward':self.wr*chase_dir_reward,'dist_reward':self.wp*dist_reward,"alive":self._alive,"angle":self.wa*angle_reward}
        return min(max(-5,total_reward),5),info
    
    def _get_action_space(self):
#         low = np.array([-13,-13,-13,-13])
#         high=np.array([13,13,13,13])
        low = np.array([-13])
        high=np.array([13])
        # MAGNITUDE AND PERIOD
#         low = np.array([0,0.2,0,0.2])
#         high=np.array([0.52,1.0,0.52,1.0])
#         low = np.array([0,0.2,0,0.2])
#         high=np.array([0.52,1.0,0.52,1.0])
        self.action_space_mean = (low+high)/2
        self.action_space_std = (high-low)/2
        return spaces.Box(low = -1,high=1,shape=low.shape)
    
    def _normalize_action(self,action):
        return np.clip((action-self.action_space_mean)/self.action_space_std,-1,1)
    def _unnormalize_action(self,action):
        action = np.clip(action,-1,1)
        return action*self.action_space_std+self.action_space_mean
#     def _apply_action(self,action):

        
        
        
# #         dynamics.setAngleCommands([force],self.rigid_data.rigidWorld.getTimestep())
    def getRotationMatrix(self,forward_vec,up_vec=np.array([0,1,0])):
        rz = forward_vec/np.linalg.norm(forward_vec)
        rx = np.cross(rz,up_vec)
        rx = rx/np.linalg.norm(rx)
        ry = np.cross(rz,rx)
        mat = np.transpose(np.array([rx,ry,rz]))
        mat.real[abs(mat.real) < 0.0001] = 0.0
        return mat
        
    
    def calc_state(self):
        
        dynamics = self.rigid_data.skeletons[0].dynamics
        
        self.body_xyz =  dynamics.getBaseLink().getPosition()
        self.body_ori =  dynamics.getBaseLinkFwd()
       

      
#         subgoal_reached = (np.linalg.norm(self.body_xyz-self.goal_pos)<0.05)
#         self.curProgress = self.path_data.trajectory.getReferencePose(self.body_xyz)
#         if subgoal_reached:
#             targetPose = self.path_data.trajectory.getPose(0.05+self.curProgress)
#             self.goal_pos = targetPose.getPosition()
#             self.goal_dir = targetPose.getOrientation()
            
        
          
        self.walk_target_dist = np.linalg.norm(self.body_xyz-self.goal_pos)

        self.walk_target_dir = self.goal_pos-self.body_xyz
        self.walk_target_dir =  self.walk_target_dir / np. linalg.norm(self.walk_target_dir)
        
        angle_to_target = np.arccos(np.dot(self.body_ori, self.walk_target_dir))
        
        
        self.angle_to_dir= np.arccos(np.dot(self.rigid_data.skeletons[0].dynamics.getBaseLinkFwd(),self.goal_dir))
        
        
        rela_vec_to_init = self.body_xyz-self.init_pos
        path_dir = self.goal_pos-self.init_pos
        path_dir = path_dir/np.linalg.norm(path_dir)
        dist_to_path = np.linalg.norm(rela_vec_to_init-path_dir*np.dot(rela_vec_to_init,path_dir))
        
#         dist_to_path = np.linalg.norm(self.path_data.trajectory.getPose(self.curProgress).getPosition()-self.body_xyz)
        if dist_to_path<0.5:
            self._alive=0
        else:
            self._alive=-1
        
        more = np.array(
            [
                np.sin(angle_to_target),
                np.cos(angle_to_target),
#                 0.3 * vx,
#                 0.3 * vy,
#                 0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
#                 r,
#                 p
            ],
            dtype=np.float32)
        return more
        
        
    def _get_obs(self):
        # obs
        
        dynamics = self.rigid_data.skeletons[0].dynamics
        more = self.calc_state()
#         joint_names = [
# #             "head",
# #                        "spine",
#             "spine02",
#                       ]
#         # # joint positions
#         joint_pos = np.concatenate([dynamics.getJoint(jnt_name).getPositions() for  jnt_name in joint_names],axis = 0)
#         joint_vel = np.concatenate([dynamics.getJoint(jnt_name).getVelocities() for  jnt_name in joint_names],axis = 0)
        joint_pos = dynamics.getPositions(includeBase=False)
        obs = np.concatenate((
               [(self.body_xyz[0]-1)/3,(self.body_xyz[2]-3)/0.2],
                joint_pos/0.52,more
        ),axis=0)
        return obs

    def reset(self):
        self.setupDynamcis()
#         self.simulator.reset()
        startPose = self.path_data.trajectory.getPose(0.001)
        path_position_begin = startPose.getPosition()
        path_orientation_begin = startPose.getOrientation()
        skeleton_dynamics = self.rigid_data.skeletons[0].dynamics
        trajectory = self.path_data.trajectory
        skeleton_dynamics.setHead(path_position_begin,path_orientation_begin)
# #         print("start point {0}  orientation {1}".format(path_position_begin,path_orientation_begin))

        self.body_xyz =  skeleton_dynamics.getBaseLink().getPosition()
        self.init_pos = self.body_xyz
        
        self.curProgress=0.001
        self.trajectory_points=[]
        
#         targetPose = self.path_data.trajectory.getPose(0.05+self.curProgress)

#         targetPose = self.path_data.trajectory.getPose(0.98)
#         self.goal_pos = targetPose.getPosition()+np.array([0,0,0.5])
#         self.goal_dir = targetPose.getOrientation()
#         theta = np.random.uniform(-3.14/2,3.14/2)
        self.goal_dir = np.array([math.cos(self.theta),0,math.sin(self.theta)])
    
#         self.goal_dir = self.goal_pos-self.init_pos
#         self.goal_dir = self.goal_dir/np.linalg.norm(self.goal_dir)
        self.goal_pos = self.init_pos+2.0*self.goal_dir
        print(self.body_xyz,skeleton_dynamics.getBaseLinkFwd(),self.goal_pos,self.goal_dir)
        self.last_obs = self._get_obs()
        
        self.dist_potential = self.calc__dist_potential()
        self.dir_potential = self.calc__dir_potential()
        return self.last_obs
    
    
    
    def plot3d(self,title=None,fig_name=None,elev=45,azim=45):
        path_points = [
            
            self.init_pos*(1.0-t)+self.goal_pos*t for t in np.arange(0.0,1.0,1.0/100)
#             self.path_data.trajectory.getPose(t).getPosition()  for t in np.arange(0.0,1.0,1.0/100)
        
        ]
        trajectory_points = self.trajectory_points
        ax=plt.figure().add_subplot(111, projection = '3d')
        ax.set_xlim(0.5,3.5)
#         ax.set_xlim(-2.5,0.5)
        ax.set_zlim(1,3)
        ax.set_ylim(1,4)
        if path_points!=None:
#             ax.scatter3D(xs=[x.data[0] for x in path_points], zs=[x.data[1] for x in path_points], ys=[x.data[2] for x in path_points],c='g')
            ax.scatter3D(xs=[x[0] for x in path_points], zs=[x[1] for x in path_points], ys=[x[2] for x in path_points],c='g')
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
        # plt.show()
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
        
    def save_data_framRate(self, save_fluid=False, save_objects=True,framRate = 10):
        if self.simulator.getIterNum()%self.simulator.getIterPerSave(framRate)!=0:
            return
        frame_num=(self.simulator.getIterNum() / self.simulator.getIterPerSave(framRate))
        self.save_data(save_fluid, save_objects,frame_num)
