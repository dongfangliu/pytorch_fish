import numpy as np
import torch as th


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

#         self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        
#         self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
#         self.rewards = np.empty((capacity, 1), dtype=np.float32)
#         self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        
        
        self.observations  = np.zeros((self.capacity, 1) + obs_shape, dtype=obs_dtype)
        self.next_observations  = np.zeros((self.capacity, 1) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self.capacity, 1,np.prod(action_shape)), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
#         np.copyto(self.obses[self.idx], obs)
#         np.copyto(self.actions[self.idx], action)
#         np.copyto(self.rewards[self.idx], reward)
#         np.copyto(self.next_obses[self.idx], next_obs)
#         np.copyto(self.not_dones[self.idx], not done)
        self.observations[self.idx] = np.array(obs).copy()
        self.next_observations[self.idx] = np.array(next_obs).copy()
        self.actions[self.idx] = np.array(action).copy()
        self.rewards[self.idx] = np.array(reward).copy()
        self.dones[self.idx] = np.array(done).copy()

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        batch_inds = np.random.randint(0,self.capacity if self.full else self.idx,size=batch_size)
        obses = th.as_tensor(self.observations[batch_inds, 0, :]).to(self.device)
        actions = th.as_tensor(self.actions[batch_inds, 0, :]).to(self.device)
        next_obses =th.as_tensor( self.next_observations[batch_inds, 0, :]).to(self.device)
        dones = th.as_tensor(self.dones[batch_inds]).to(self.device)
        rewards = th.as_tensor(self.rewards[batch_inds]).to(self.device)

#         obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
#         actions = torch.as_tensor(self.actions[idxs], device=self.device)
#         rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
#         next_obses = torch.as_tensor(self.next_obses[idxs],
#                                      device=self.device).float()
#         not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, dones