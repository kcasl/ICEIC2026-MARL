"""네 채널 신호판.

참고한 구조는 세 가지를 겹친 것이다.

- Foerster et al., 2016, Learning to Communicate with Deep Multi-Agent RL (RIAL/DIAL).
  수신자는 상대가 고른 이산 기호를 관측한다. PPO는 관측을 통해 기울기를
  직접 흘리지 못하므로, 기호 선택은 RIAL처럼 각 에이전트의 행동으로 학습한다.
  MAPPO의 중앙 크리틱이 두 관측을 같이 보므로, 어떤 기호가 팀 보상에
  도움이 됐는지는 크리틱 쪽에서 신용이 갈린다.
- Singh et al., 2019, IC3Net. 보낼지 말지를 게이트로 고른다.
- 슬롯은 한 스텝짜리 메시지가 아니다. 채널에 올린 기호는 같은 채널에
  다른 기호가 써질 때까지 남아 있다. 상대는 매 스텝 네 채널을 전부 듣는다.

기호 0은 빈 칸이다. 1부터 7은 미리 뜻을 붙이지 않는다.
"""

from dataclasses import dataclass, field

import numpy as np

from marl.config import N_CHANNELS, VOCAB


@dataclass
class Slot:
    symbol: int = 0
    speaker: int = 0  # 0 없음, 1 또는 2


@dataclass
class Blackboard:
    slots: list[Slot] = field(default_factory=lambda: [Slot() for _ in range(N_CHANNELS)])

    def reset(self) -> None:
        self.slots = [Slot() for _ in range(N_CHANNELS)]

    def write(self, speaker: int, channel: int, symbol: int) -> None:
        """게이트가 열린 행동만 이 함수로 들어온다."""
        slot = self.slots[int(channel) % N_CHANNELS]
        slot.symbol = int(symbol) % VOCAB
        slot.speaker = 0 if slot.symbol == 0 else int(speaker)

    def features(self) -> np.ndarray:
        """네 채널을 이어 붙인 벡터. 기호 원핫 8 + 화자 원핫 3."""
        parts = []
        for slot in self.slots:
            symbol = np.zeros(VOCAB, dtype=np.float32)
            symbol[slot.symbol] = 1.0
            speaker = np.zeros(3, dtype=np.float32)
            speaker[slot.speaker] = 1.0
            parts.append(symbol)
            parts.append(speaker)
        return np.concatenate(parts)
