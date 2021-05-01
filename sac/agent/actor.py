from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import math
from torch import nn
import torch.nn.functional as F
from .distributions import *
from .utils import create_mlp


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, action_dim, net_arch,log_std_bounds,
        activation_fn: Type[nn.Module] = nn.ReLU):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        latent_pi_net = create_mlp(obs_dim, -1, net_arch,activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else obs_dim
        self.mu = nn.Linear(last_layer_dim, action_dim)
        self.log_std = nn.Linear(last_layer_dim, action_dim)
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        
    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        x = self.latent_pi(obs)
        mu = self.mu(x)
        log_std = self.log_std(x)
        # constrain log_std inside [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clamp(log_std, log_std_min, log_std_max)
        return mu,log_std,{}
    
    def _predict(self,obs,deterministic = False):
        return self.forward(obs, deterministic)
        
    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

        
    def forward(self, obs,deterministic = False):
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)
