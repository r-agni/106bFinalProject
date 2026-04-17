# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import math
from torch.optim import lr_scheduler

from torchrl.data import Composite, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union
import einops

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import GAE

@dataclass
class PPOConfig:
    name: str = "ppo"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    clip_param: float = 0.1
    entropy_coef: float = 0.001
    gae_lambda: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 5.0

    # whether to use privileged information
    priv_actor: bool = False
    priv_critic: bool = False

    checkpoint_path: Union[str, None] = None
    adaptive_entropy: dict = field(default_factory=dict)
    actor: dict = field(default_factory=lambda: {
        "lr": 5e-4,
        "lr_scheduler": None,
        "lr_scheduler_kwargs": {},
        "hidden_units": [256, 256, 256],
        "activation": "leaky_relu",
        "layer_norm": True,
        "weight_decay": 0.0,
    })
    critic: dict = field(default_factory=lambda: {
        "lr": 5e-4,
        "lr_scheduler": None,
        "lr_scheduler_kwargs": {},
        "hidden_units": [256, 256, 256],
        "activation": "leaky_relu",
        "layer_norm": True,
        "weight_decay": 0.0,
        "use_huber_loss": True,
        "huber_delta": 10.0,
    })

cs = ConfigStore.instance()
cs.store("ppo", node=PPOConfig, group="algo")
cs.store("ppo_priv", node=PPOConfig(priv_actor=True, priv_critic=True), group="algo")
cs.store("ppo_priv_critic", node=PPOConfig(priv_critic=True), group="algo")


def _cfg_get(container, key, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _make_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    key = str(name).lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation '{name}'. Supported: {list(activations.keys())}")
    return activations[key]()


def make_mlp(num_units, activation: str = "leaky_relu", layer_norm: bool = True):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(_make_activation(activation))
        if layer_norm:
            layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


def _resolve_scheduler(name: str):
    if name is None or str(name).strip() == "":
        return None
    name = str(name)
    if name.startswith("lr_scheduler."):
        name = name.split(".", 1)[1]
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    raise ValueError(f"Unknown lr scheduler: {name}")


class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale


class FlattenCat(nn.Module):
    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        normalized_tensors = []
        for tensor in tensors:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1).unsqueeze(-1)
            elif tensor.ndim == 2:
                tensor = tensor.unsqueeze(-1)
            normalized_tensors.append(tensor)
        return torch.cat(normalized_tensors, dim=-1)


class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = cfg.entropy_coef
        self.clip_param = cfg.clip_param
        self.max_grad_norm = cfg.max_grad_norm
        self.n_agents, self.action_dim = action_spec[("agents", "action")].shape[-2:]
        self.gae = GAE(cfg.gamma, cfg.gae_lambda)

        actor_cfg = cfg.actor
        critic_cfg = cfg.critic
        actor_hidden_units = _cfg_get(actor_cfg, "hidden_units", [256, 256, 256])
        actor_activation = _cfg_get(actor_cfg, "activation", "leaky_relu")
        actor_layer_norm = _cfg_get(actor_cfg, "layer_norm", True)
        critic_hidden_units = _cfg_get(critic_cfg, "hidden_units", [256, 256, 256])
        critic_activation = _cfg_get(critic_cfg, "activation", "leaky_relu")
        critic_layer_norm = _cfg_get(critic_cfg, "layer_norm", True)
        actor_lr = _cfg_get(actor_cfg, "lr", 5e-4)
        critic_lr = _cfg_get(critic_cfg, "lr", 5e-4)
        actor_weight_decay = _cfg_get(actor_cfg, "weight_decay", 0.0)
        critic_weight_decay = _cfg_get(critic_cfg, "weight_decay", 0.0)
        critic_use_huber_loss = _cfg_get(critic_cfg, "use_huber_loss", True)
        critic_huber_delta = _cfg_get(critic_cfg, "huber_delta", 10.0)

        if critic_use_huber_loss:
            self.critic_loss_fn = nn.HuberLoss(delta=critic_huber_delta)
        else:
            self.critic_loss_fn = nn.MSELoss()

        fake_input = observation_spec.zero()
        intrinsics_prefix = ("agents", "intrinsics")
        intrinsics_keys = [
            key for key in observation_spec.keys(True, True)
            if isinstance(key, tuple) and key[:len(intrinsics_prefix)] == intrinsics_prefix and key != intrinsics_prefix
        ]

        if self.cfg.priv_actor:
            if intrinsics_keys:
                actor_context_module = TensorDictSequential(
                    TensorDictModule(FlattenCat(), intrinsics_keys, ["_intrinsics_flat"]),
                    TensorDictModule(
                        make_mlp([64, 64], activation=actor_activation, layer_norm=actor_layer_norm),
                        ["_intrinsics_flat"],
                        ["context"],
                    ),
                )
            else:
                intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
                actor_context_module = TensorDictModule(
                    nn.Sequential(
                        nn.LayerNorm(intrinsics_dim),
                        make_mlp([64, 64], activation=actor_activation, layer_norm=actor_layer_norm),
                    ),
                    [("agents", "intrinsics")],
                    ["context"],
                )
            actor_module = TensorDictSequential(
                TensorDictModule(
                    make_mlp([128, 128], activation=actor_activation, layer_norm=actor_layer_norm),
                    [("agents", "observation")],
                    ["feature"],
                ),
                actor_context_module,
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(
                        make_mlp(actor_hidden_units, activation=actor_activation, layer_norm=actor_layer_norm),
                        Actor(self.action_dim),
                    ),
                    ["feature"],
                    ["loc", "scale"],
                ),
            )
        else:
            actor_module=TensorDictModule(
                nn.Sequential(
                    make_mlp(actor_hidden_units, activation=actor_activation, layer_norm=actor_layer_norm),
                    Actor(self.action_dim),
                ),
                [("agents", "observation")], ["loc", "scale"]
            )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
            log_prob_key="sample_log_prob"
        ).to(self.device)

        if self.cfg.priv_critic:
            if intrinsics_keys:
                critic_context_module = TensorDictSequential(
                    TensorDictModule(FlattenCat(), intrinsics_keys, ["_intrinsics_flat"]),
                    TensorDictModule(
                        make_mlp([64, 64], activation=critic_activation, layer_norm=critic_layer_norm),
                        ["_intrinsics_flat"],
                        ["context"],
                    ),
                )
            else:
                intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
                critic_context_module = TensorDictModule(
                    nn.Sequential(
                        nn.LayerNorm(intrinsics_dim),
                        make_mlp([64, 64], activation=critic_activation, layer_norm=critic_layer_norm),
                    ),
                    [("agents", "intrinsics")],
                    ["context"],
                )
            self.critic = TensorDictSequential(
                TensorDictModule(
                    make_mlp([128, 128], activation=critic_activation, layer_norm=critic_layer_norm),
                    [("agents", "observation")],
                    ["feature"],
                ),
                critic_context_module,
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(
                        make_mlp(critic_hidden_units, activation=critic_activation, layer_norm=critic_layer_norm),
                        nn.LazyLinear(1),
                    ),
                    ["feature"],
                    ["state_value"],
                )
            ).to(self.device)
        else:
            self.critic = TensorDictModule(
                nn.Sequential(
                    make_mlp(critic_hidden_units, activation=critic_activation, layer_norm=critic_layer_norm),
                    nn.LazyLinear(1),
                ),
                [("agents", "observation")], ["state_value"]
            ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            print(f"\n\n--------------------")
            print("No model loaded, using an random initial policy")
            print(f"--------------------\n\n")

            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)

            self.actor.apply(init_)
            self.critic.apply(init_)

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
            weight_decay=actor_weight_decay,
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
            weight_decay=critic_weight_decay,
        )
        actor_scheduler_cls = _resolve_scheduler(_cfg_get(actor_cfg, "lr_scheduler", None))
        critic_scheduler_cls = _resolve_scheduler(_cfg_get(critic_cfg, "lr_scheduler", None))
        self.actor_opt_scheduler = None
        self.critic_opt_scheduler = None
        if actor_scheduler_cls is not None:
            self.actor_opt_scheduler = actor_scheduler_cls(
                self.actor_opt, **(_cfg_get(actor_cfg, "lr_scheduler_kwargs", {}) or {})
            )
        if critic_scheduler_cls is not None:
            self.critic_opt_scheduler = critic_scheduler_cls(
                self.critic_opt, **(_cfg_get(critic_cfg, "lr_scheduler_kwargs", {}) or {})
            )
        self.value_norm = ValueNorm1(reward_spec[("agents", "reward")].shape[-2:]).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "feature", "context", "_intrinsics_flat", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        dones = einops.repeat(
            tensordict[("next", "terminated")],
            "t e 1 -> t e a 1",
            a=self.n_agents
        )
        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        if self.actor_opt_scheduler is not None:
            self.actor_opt_scheduler.step()
        if self.critic_opt_scheduler is not None:
            self.critic_opt_scheduler.step()

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        out = {k: v.item() for k, v in infos.items()}
        out["actor_lr"] = self.actor_opt.param_groups[0]["lr"]
        out["critic_lr"] = self.critic_opt.param_groups[0]["lr"]
        return out

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        loss = policy_loss + entropy_loss + value_loss
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.actor_opt.step()
        self.critic_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
