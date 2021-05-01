import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

from . import utils
from .actor import *
import copy
class DiagGaussianActor_HR(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim,action_dim, hidden_dim, hidden_depth,
                 log_std_bounds,base_policy_param_dicts =[]):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.gate_network = utils.mlp(obs_dim, hidden_dim, len(base_policy_param_dicts),hidden_depth,output_mod = nn.Softmax(dim=-1))
        self.hidden_depth=hidden_depth
        policy_num=len(base_policy_param_dicts)
        self.w_params = nn.ParameterList([
                nn.Parameter(torch.stack( [ base_policy_param_dicts[i]['trunk.{}.weight'.format(d*2)]  for i in range(policy_num)],dim = 0),requires_grad= False) for d in range(hidden_depth+1)
                            ])
        self.bias_params= nn.ParameterList([
                nn.Parameter(torch.stack( [ base_policy_param_dicts[i]['trunk.{}.bias'.format(d*2)]  for i in range(policy_num)],dim = 0),requires_grad= False) for d in range(hidden_depth+1)
                            ])
        
        self.outputs = dict()
        self.apply(utils.weight_init)
        
    def buildDNN(self,blend_params,obs):
        w_param_new = [torch.reshape(torch.matmul(blend_params,torch.flatten(w,start_dim=1)),(-1,*w.shape[1:])) for w in self.w_params]
        bias_param_new = [torch.reshape(torch.matmul(blend_params,torch.flatten(b,start_dim=1)),(-1,*b.shape[1:])) for b in self.bias_params]
        mods = torch.unsqueeze(obs,-1)
        if self.hidden_depth == 0:
            mods = torch.bmm(w_param_new[i],mods)+torch.unsqueeze(bias_param_new[i],-1)
        else:
            for i in range(self.hidden_depth):
                mods=F.relu(torch.bmm(w_param_new[i],mods)+torch.unsqueeze(bias_param_new[i],-1))
            mods=torch.bmm(w_param_new[self.hidden_depth],mods)+torch.unsqueeze(bias_param_new[self.hidden_depth],-1)
        return torch.squeeze(mods,-1)
        
    def forward(self, obs):
        blend_params = self.gate_network(obs)
        mu, log_std =self.buildDNN(blend_params,obs).chunk(2, dim=-1)
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        
        return dist


    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)