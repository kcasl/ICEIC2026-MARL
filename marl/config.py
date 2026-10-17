"""학습과 환경이 같이 쓰는 상수.

PPO 기본값은 Yu et al., MAPPO(NeurIPS 2022)와 Schulman et al., PPO(2017)를
격자 탈출에 맞게 줄인 것이다. 롤아웃이 짧아도 통신 행동이 사라지지 않도록
이산 행동의 엔트로피 계수는 조금 크게 둔다.
"""

from dataclasses import dataclass


# 한 칸의 길이를 1로 둔다. 에이전트 지름은 약 0.64라서 세 칸 통로에 둘이 나란히 선다.
AGENT_RADIUS = 0.32
OBJECT_RADIUS = 0.36
# 발판/도착 판정. 칸 중심에서 이 거리 안이면 올라선 것이다.
TRIGGER_RADIUS = 0.55

# 정면으로만 전진한다. 후진은 없고, 세타(각속도)로 방향을 튼다.
V_MAX = 2.5
W_MAX = 2.2
DT = 0.2

# 둘이 이 거리 안에 있고 둘 다 전진해야 물건이 움직인다.
ATTACH_GAP = 0.1
PUSH_SPEED = 0.25

N_CHANNELS = 4
# 0은 빈 슬롯. 1..7은 의미가 정해져 있지 않고 학습으로 역할이 생긴다.
VOCAB = 8

# 관측을 고정 길이로 맞추기 위한 상한. 생성기가 이 개수를 넘기지 않는다.
MAX_PLATES = 2
MAX_DOORS = 2
MAX_OBJECTS = 2
MAX_GOALS = 2


@dataclass
class TrainConfig:
    """MAPPO가 기본이고, 학습이 안 되면 --algo ippo 로 독립 PPO를 쓴다."""

    algo: str = "mappo"
    seed: int = 0
    updates: int = 200
    rollout_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatch: int = 64
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    episode_steps: int = 400
    # 잠재함수 셰이핑 계수. 1이면 Ng et al. 1999 그대로이고, 최적 정책은 바뀌지 않는다.
    shaping_coef: float = 1.0
    time_penalty: float = 0.001
    success_bonus: float = 1.0
    maps_dir: str = "maps/train"
    device: str = "cpu"
