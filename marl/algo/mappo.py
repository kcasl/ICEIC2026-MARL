"""MAPPO와, 학습이 안 될 때의 대안으로 IPPO.

MAPPO (Yu et al., NeurIPS 2022): 액터는 자기 관측만 보고, 크리틱은 두 관측을 같이 본다.
에이전트 둘의 파라미터는 공유하지 않는다. 역할이 다르기 때문이다.
IPPO는 크리틱도 자기 관측만 본다. 중앙 크리틱이 불안정할 때 같은 보상으로 바꿀 수 있다.

보상은 팀 공유다. 어드밴티지는 GAE(Schulman et al., 2016)로 만든다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from marl.algo.network import Actor, Critic
from marl.config import TrainConfig
from marl.env.escape_env import OBS_DIM, EscapeEnv


class Trainer:
    def __init__(self, env: EscapeEnv, cfg: TrainConfig):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.actors = [Actor(OBS_DIM, cfg.hidden).to(self.device) for _ in range(2)]
        critic_dim = OBS_DIM * 2 if cfg.algo == "mappo" else OBS_DIM
        if cfg.algo == "mappo":
            self.critics = [Critic(critic_dim, cfg.hidden).to(self.device)]
        elif cfg.algo == "ippo":
            self.critics = [Critic(critic_dim, cfg.hidden).to(self.device) for _ in range(2)]
        else:
            raise ValueError(f"unknown algo {cfg.algo}")
        params = [param for actor in self.actors for param in actor.parameters()]
        params += [param for critic in self.critics for param in critic.parameters()]
        self.optim = torch.optim.Adam(params, lr=cfg.lr)

    def collect(self) -> dict:
        cfg = self.cfg
        obs = self.env.reset()
        store = {key: [] for key in ("obs", "raw", "send", "channel", "symbol", "logp", "reward", "value", "terminated", "done", "bootstrap")}
        for _ in range(cfg.rollout_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions = [actor.act(obs_t[index:index + 1]) for index, actor in enumerate(self.actors)]
                value = self._value(obs_t)
            env_actions = []
            for action in actions:
                env_actions.append({
                    "v": float(action["v"].item()),
                    "omega": float(action["omega"].item()),
                    "send": int(action["send"].item()),
                    "channel": int(action["channel"].item()),
                    "symbol": int(action["symbol"].item()),
                })
            next_obs, reward, terminated, truncated, info = self.env.step(env_actions)
            done = terminated or truncated
            with torch.no_grad():
                if terminated:
                    bootstrap = np.zeros(1 if self.cfg.algo == "mappo" else 2, dtype=np.float32)
                elif truncated:
                    bootstrap = self._value(torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)).cpu().numpy().reshape(-1)
                else:
                    bootstrap = np.zeros(1 if self.cfg.algo == "mappo" else 2, dtype=np.float32)
            store["obs"].append(obs.copy())
            store["raw"].append(np.stack([action["raw"].squeeze(0).cpu().numpy() for action in actions]))
            store["send"].append([action["send"].item() for action in actions])
            store["channel"].append([action["channel"].item() for action in actions])
            store["symbol"].append([action["symbol"].item() for action in actions])
            store["logp"].append([float(action["logp"].item()) for action in actions])
            store["reward"].append(reward)
            store["value"].append(value.detach().cpu().numpy())
            store["terminated"].append(terminated)
            store["done"].append(done)
            store["bootstrap"].append(bootstrap)
            store.setdefault("success", []).append(bool(info["success"]))
            obs = self.env.reset() if done else next_obs
        store["last_obs"] = obs
        return store

    def _value(self, obs: torch.Tensor) -> torch.Tensor:
        """MAPPO는 길이 1, IPPO는 길이 2."""
        if self.cfg.algo == "mappo":
            joint = torch.cat([obs[0], obs[1]], dim=-1).unsqueeze(0)
            return self.critics[0](joint).view(1)
        values = [critic(obs[index:index + 1]) for index, critic in enumerate(self.critics)]
        return torch.cat(values, dim=0)

    def advantages(self, batch: dict) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        rewards = np.asarray(batch["reward"], dtype=np.float32)
        values = np.asarray(batch["value"], dtype=np.float32)
        terminated = np.asarray(batch["terminated"], dtype=np.bool_)
        done = np.asarray(batch["done"], dtype=np.bool_)
        bootstrap = np.asarray(batch["bootstrap"], dtype=np.float32)
        steps = len(rewards)
        if values.ndim == 1:
            values = values[:, None]
        agents = values.shape[1]
        adv = np.zeros_like(values)
        gae = np.zeros(agents, dtype=np.float32)
        with torch.no_grad():
            last = self._value(torch.as_tensor(batch["last_obs"], dtype=torch.float32, device=self.device))
            last_np = last.cpu().numpy().reshape(-1)
        if agents == 1 and last_np.shape[0] != 1:
            last_np = np.array([float(last_np.mean())], dtype=np.float32)
        for t in range(steps - 1, -1, -1):
            if t == steps - 1:
                next_v = last_np
            else:
                next_v = values[t + 1]
            if terminated[t]:
                next_v = np.zeros(agents, dtype=np.float32)
            elif done[t]:
                # 시간 초과. 새 에피소드의 가치로 이어 붙이지 않고, 끊긴 상태의 가치만 쓴다.
                next_v = np.asarray(bootstrap[t], dtype=np.float32).reshape(-1)
                if next_v.shape[0] == 1 and agents == 1:
                    next_v = next_v
                elif next_v.shape[0] != agents:
                    next_v = np.full(agents, float(next_v.mean()), dtype=np.float32)
            delta = rewards[t] + cfg.gamma * next_v - values[t]
            cont = 0.0 if done[t] else 1.0
            gae = delta + cfg.gamma * cfg.gae_lambda * cont * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    def update(self, batch: dict) -> dict:
        cfg = self.cfg
        adv, returns = self.advantages(batch)
        obs = np.asarray(batch["obs"], dtype=np.float32)
        raw = np.asarray(batch["raw"], dtype=np.float32)
        send = np.asarray(batch["send"], dtype=np.int64)
        channel = np.asarray(batch["channel"], dtype=np.int64)
        symbol = np.asarray(batch["symbol"], dtype=np.int64)
        old_logp = np.asarray(batch["logp"], dtype=np.float32)
        steps = obs.shape[0]
        # MAPPO 어드밴티지는 팀 하나. IPPO는 에이전트별.
        if adv.shape[1] == 1:
            adv_use = np.repeat(adv, 2, axis=1)
            ret_use = np.repeat(returns, 2, axis=1)
        else:
            adv_use = adv
            ret_use = returns
        flat_adv = adv_use.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        adv_use = flat_adv.reshape(steps, 2)

        indices = np.arange(steps)
        stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "count": 0}
        for _ in range(cfg.epochs):
            self.rng_shuffle(indices)
            for start in range(0, steps, cfg.minibatch):
                mb = indices[start:start + cfg.minibatch]
                self.optim.zero_grad()
                policy_loss = 0.0
                entropy_loss = 0.0
                for agent in range(2):
                    obs_t = torch.as_tensor(obs[mb, agent], device=self.device)
                    raw_t = torch.as_tensor(raw[mb, agent], device=self.device)
                    logp, entropy = self.actors[agent].evaluate(
                        obs_t,
                        raw_t,
                        torch.as_tensor(send[mb, agent], device=self.device),
                        torch.as_tensor(channel[mb, agent], device=self.device),
                        torch.as_tensor(symbol[mb, agent], device=self.device),
                    )
                    old = torch.as_tensor(old_logp[mb, agent], device=self.device)
                    advantage = torch.as_tensor(adv_use[mb, agent], device=self.device)
                    ratio = torch.exp(logp - old)
                    clipped = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * advantage
                    policy_loss = policy_loss + -torch.min(ratio * advantage, clipped).mean()
                    entropy_loss = entropy_loss + entropy.mean()
                value_loss = self._value_loss(obs[mb], ret_use[mb])
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [param for actor in self.actors for param in actor.parameters()]
                    + [param for critic in self.critics for param in critic.parameters()],
                    cfg.max_grad_norm,
                )
                self.optim.step()
                stats["policy"] += float(policy_loss.detach())
                stats["value"] += float(value_loss.detach())
                stats["entropy"] += float(entropy_loss.detach())
                stats["count"] += 1
        count = max(stats["count"], 1)
        return {
            "policy_loss": stats["policy"] / count,
            "value_loss": stats["value"] / count,
            "entropy": stats["entropy"] / count,
            "return": float(np.mean(batch["reward"])),
            "success": float(np.mean(batch["success"])),
        }

    def _value_loss(self, obs: np.ndarray, returns: np.ndarray) -> torch.Tensor:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if self.cfg.algo == "mappo":
            joint = torch.cat([obs_t[:, 0], obs_t[:, 1]], dim=-1)
            pred = self.critics[0](joint)
            target = torch.as_tensor(returns[:, 0], dtype=torch.float32, device=self.device)
            return (pred - target).pow(2).mean()
        loss = 0.0
        for agent, critic in enumerate(self.critics):
            pred = critic(obs_t[:, agent])
            target = torch.as_tensor(returns[:, agent], dtype=torch.float32, device=self.device)
            loss = loss + (pred - target).pow(2).mean()
        return loss

    @staticmethod
    def rng_shuffle(indices: np.ndarray) -> None:
        np.random.shuffle(indices)
