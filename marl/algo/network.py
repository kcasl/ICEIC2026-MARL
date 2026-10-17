"""에이전트마다 따로 있는 액터와, 팀을 보는 크리틱.

연속 행동 (전진 속력, 각속도)은 tanh 가우시안이다.
통신은 이산 머리 세 개다. 보낼지, 어느 채널에, 어떤 기호를.
크리틱은 MAPPO일 때 두 관측을 이어 붙인 입력을 보고, IPPO일 때는 자기 관측만 본다.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from marl.config import N_CHANNELS, V_MAX, VOCAB, W_MAX


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, 2)
        # 초반 탐색이 너무 작지 않게 표준편차를 0.5 근처에서 시작한다.
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        self.gate = nn.Linear(hidden, 2)
        self.channel = nn.Linear(hidden, N_CHANNELS)
        self.symbol = nn.Linear(hidden, VOCAB)

    def _heads(self, obs: torch.Tensor):
        hidden = self.body(obs)
        std = self.log_std.exp().clamp(1e-4, 2.0).expand_as(self.mu(hidden))
        return self.mu(hidden), std, self.gate(hidden), self.channel(hidden), self.symbol(hidden)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mu, std, gate_logits, channel_logits, symbol_logits = self._heads(obs)
        normal = Normal(mu, std)
        raw = mu if deterministic else normal.rsample()
        logp, speed, omega, entropy = self._continuous(normal, raw)
        gate_dist = Categorical(logits=gate_logits)
        channel_dist = Categorical(logits=channel_logits)
        symbol_dist = Categorical(logits=symbol_logits)
        if deterministic:
            gate = torch.argmax(gate_logits, dim=-1)
            channel = torch.argmax(channel_logits, dim=-1)
            symbol = torch.argmax(symbol_logits, dim=-1)
        else:
            gate = gate_dist.sample()
            channel = channel_dist.sample()
            symbol = symbol_dist.sample()
        logp = logp + gate_dist.log_prob(gate) + channel_dist.log_prob(channel) + symbol_dist.log_prob(symbol)
        entropy = entropy + gate_dist.entropy() + channel_dist.entropy() + symbol_dist.entropy()
        return {
            "raw": raw,
            "v": speed,
            "omega": omega,
            "send": gate,
            "channel": channel,
            "symbol": symbol,
            "logp": logp,
            "entropy": entropy,
        }

    def evaluate(self, obs: torch.Tensor, raw: torch.Tensor, send, channel, symbol):
        mu, std, gate_logits, channel_logits, symbol_logits = self._heads(obs)
        normal = Normal(mu, std)
        logp, _, _, entropy = self._continuous(normal, raw)
        gate_dist = Categorical(logits=gate_logits)
        channel_dist = Categorical(logits=channel_logits)
        symbol_dist = Categorical(logits=symbol_logits)
        logp = logp + gate_dist.log_prob(send) + channel_dist.log_prob(channel) + symbol_dist.log_prob(symbol)
        entropy = entropy + gate_dist.entropy() + channel_dist.entropy() + symbol_dist.entropy()
        return logp, entropy

    @staticmethod
    def _continuous(normal: Normal, raw: torch.Tensor):
        squashed = torch.tanh(raw)
        # tanh 보정 뒤에, 속력은 (0, V_MAX), 각속도는 (-W_MAX, W_MAX)로 펼친다.
        logp = normal.log_prob(raw).sum(-1)
        logp = logp - torch.log(1.0 - squashed.pow(2) + 1e-6).sum(-1)
        logp = logp - math.log(V_MAX * 0.5) - math.log(W_MAX)
        speed = (squashed[..., 0] + 1.0) * 0.5 * V_MAX
        omega = squashed[..., 1] * W_MAX
        return logp, speed, omega, normal.entropy().sum(-1)


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = _mlp(in_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
