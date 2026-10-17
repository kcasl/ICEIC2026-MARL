"""협력 탈출 학습 진입점.

    python -m marl.train --check
    python -m marl.train --updates 200
    python -m marl.train --algo ippo

MAPPO가 기본이다. 성공률이 오래 0이면 IPPO로 다시 돌린다.
IPPO는 중앙 크리틱 없이 각 에이전트가 자기 관측만으로 가치를 매긴다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from marl.algo.mappo import Trainer
from marl.config import TrainConfig
from marl.env.escape_env import EscapeEnv, run_mechanics_checks


def build_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    cfg.algo = args.algo
    cfg.seed = args.seed
    cfg.updates = args.updates
    cfg.maps_dir = args.maps
    cfg.device = args.device
    if args.rollout is not None:
        cfg.rollout_steps = args.rollout
    return cfg


def train(cfg: TrainConfig) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = EscapeEnv(cfg.maps_dir, cfg)
    trainer = Trainer(env, cfg)
    print(f"algo={cfg.algo} maps={len(env.folders)} device={cfg.device}")
    best = -1e9
    stall = 0
    for update in range(1, cfg.updates + 1):
        batch = trainer.collect()
        stats = trainer.update(batch)
        best = max(best, stats["success"])
        stall = 0 if stats["success"] > 0 else stall + 1
        print(
            f"update {update:04d}  reward {stats['return']:.4f}  "
            f"success {stats['success']:.3f}  policy {stats['policy_loss']:.3f}  "
            f"value {stats['value_loss']:.3f}  entropy {stats['entropy']:.3f}"
        )
        if cfg.algo == "mappo" and stall == 30 and update >= 30:
            print("30회 동안 성공이 없다. 같은 설정으로 --algo ippo 를 다시 실행한다.")
            stall = 10**9
        if cfg.updates >= 30 and best <= 0 and cfg.algo == "mappo":
            print("이번 실행에서는 탈출이 없었다. 대안으로 python -m marl.train --algo ippo")


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description="2-agent continuous escape with 4-channel communication")
    parser.add_argument("--algo", choices=("mappo", "ippo"), default="mappo")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maps", default="maps/train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check", action="store_true", help="물리와 통신만 확인하고 끝낸다")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    maps = root / args.maps if not Path(args.maps).is_absolute() else Path(args.maps)
    if args.check:
        run_mechanics_checks(maps)
        return
    cfg = build_config(args)
    cfg.maps_dir = str(maps)
    train(cfg)


if __name__ == "__main__":
    main()
