from typing import Any, Dict, List, Optional, Tuple, Type, Union,Iterable

import numpy as np
import torch as th
import torch.nn.functional as F

from . import Agent, utils

from itertools import zip_longest
from .actor import *
def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.
    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo
def polyak_update(params: Iterable[th.nn.Parameter], target_params: Iterable[th.nn.Parameter], tau: float) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93
    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with th.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)
class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_space,device, 
                 critic_network,critic_target_network,
                 actor_network,
                 replay_buffer, 
                 discount=0.99, 
                 ent_coef: Union[str, float] = "auto", 
                 target_entropy: Union[str, float] = "auto",
                 alpha_lr=3e-4, 
                 alpha_betas=[0.9, 0.999],
                 actor_lr=3e-4, 
                 actor_betas=[0.9, 0.999], 
                 critic_lr=3e-4, 
                 critic_betas=[0.9, 0.999], 
                 critic_tau=0.005, 
                 gradient_steps=1,
                 target_update_interval=1,
                 batch_size=64):
        super().__init__()

        self.device = th.device(device)
        self.action_space = action_space
        self.discount = discount
        self.critic_tau = critic_tau
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size

        self.critic = critic_network.to(self.device)

        self.critic_target = critic_target_network.to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = actor_network.to(self.device)

        self.replay_buffer = replay_buffer
        self._n_updates = 0
        
        ## entropy related
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.target_entropy = target_entropy
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -action_dim
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=alpha_lr)
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

        # optimizers
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)

        self.critic_optimizer = th.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)



    def save(self,model_path):
        th.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'ent_coef_optimizer_state_dict': self.ent_coef_optimizer.state_dict(),
            'log_ent_coef':self.log_ent_coef,
#             'replay_buffer':self.replay_buffer
        }, model_path)

    def load(self,model_path):
        checkpoint = th.load(model_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.ent_coef_optimizer.load_state_dict(checkpoint['ent_coef_optimizer_state_dict'])
        self.log_ent_coef =checkpoint['log_ent_coef']
#         self.replay_buffer = checkpoint['replay_buffer']
    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)
        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))


    def predict(self, obs, deterministic: bool = False):
        obs = th.as_tensor(obs).to(self.device).float()
        with th.no_grad():
            actions = self.actor(obs, deterministic)
        actions = self.unscale_action(actions.cpu().numpy())
        return actions


    def train(self,gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor_optimizer, self.critic_optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
#         self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(next_obs, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = reward + (1-done) * self.discount * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(obs, action)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(obs, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.critic_tau)

        self._n_updates += gradient_steps

        log_data = {}
        log_data['train/batch_reward']= reward.mean()
        log_data["train/n_updates"] = self._n_updates
        log_data["train/ent_coef"] = np.mean(ent_coefs)
        log_data["train/target_entropy"]=self.target_entropy
        log_data["train/actor_loss"] = np.mean(actor_losses)
        log_data["train/critic_loss"] = np.mean(critic_losses)
        if len(ent_coef_losses) > 0:
            log_data["train/ent_coef_loss"] = np.mean(ent_coef_losses)
        return log_data