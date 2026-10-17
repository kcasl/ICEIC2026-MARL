"""연속 공간 협력 탈출.

맵 YAML의 벽은 그대로 쓰고, 에이전트는 칸에 묶이지 않는다.
행동은 각속도 ω로 세타를 바꾼 뒤, 바라보는 방향으로만 속력 v만큼 전진한다.
두 에이전트의 몸통은 겹칠 수 있다. 벽, 닫힌 문, 물건은 겹치지 못한다.

물건은 두 에이전트가 동시에 붙어 있고, 둘 다 전진할 때만 움직인다.
한 명만 붙어 있으면 물건은 벽처럼 막기만 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from marl.config import (
    AGENT_RADIUS,
    ATTACH_GAP,
    DT,
    MAX_DOORS,
    MAX_GOALS,
    MAX_OBJECTS,
    MAX_PLATES,
    N_CHANNELS,
    OBJECT_RADIUS,
    PUSH_SPEED,
    TRIGGER_RADIUS,
    VOCAB,
    TrainConfig,
)
from marl.env.communication import Blackboard
from marl.env.reward import potential, team_reward


def _center(cell_x: int, cell_y: int) -> tuple[float, float]:
    return cell_x + 0.5, cell_y + 0.5


@dataclass
class Body:
    x: float
    y: float
    theta: float


@dataclass
class Plate:
    id: str
    x: float
    y: float
    opens: list[str]
    mode: str
    owner: str | None = None
    latched: bool = False
    occupied: bool = False

    @property
    def active(self) -> bool:
        # latch는 한 번 밟으면 유지, hold는 지금 위에 있을 때만 켜진다.
        return self.latched if self.mode == "latch" else self.occupied


@dataclass
class Door:
    id: str
    cells: list[tuple[int, int]]
    opened_by: list[str]
    logic: str
    mode: str
    blocks_agent: str | None = None
    open: bool = False


@dataclass
class Crate:
    id: str
    x: float
    y: float
    onto_plate: str | None


@dataclass
class Goal:
    id: str
    x: float
    y: float
    agent: str


@dataclass
class MapWorld:
    map_id: str
    map_type: str
    width: int
    height: int
    walls: set[tuple[int, int]]
    plates: list[Plate]
    doors: list[Door]
    objects: list[Crate]
    goals: list[Goal]
    starts: list[tuple[float, float]]
    independent: bool


def load_map(folder: Path) -> MapWorld:
    """맵 폴더의 YAML을 연속 좌표 세계로 읽는다."""
    walls_doc = _read(folder / "walls.yaml")
    plates_doc = _read(folder / "plates.yaml")
    doors_doc = _read(folder / "doors.yaml")
    objects_doc = _read(folder / "objects.yaml")
    goals_doc = _read(folder / "goal.yaml")
    agent1 = _read(folder / "agent1_start.yaml")
    agent2 = _read(folder / "agent2_start.yaml")
    walls = {(int(x), int(y)) for x, y in walls_doc["cells"]}
    plates = []
    for plate in plates_doc["plates"]:
        px, py = _center(int(plate["x"]), int(plate["y"]))
        plates.append(
            Plate(
                id=plate["id"],
                x=px,
                y=py,
                opens=list(plate.get("opens", [])),
                mode=plate.get("mode", plates_doc.get("mode", "hold")),
                owner=None if plate.get("pressed_by_agent") is None else str(plate["pressed_by_agent"]),
            )
        )
    doors = []
    for door in doors_doc["doors"]:
        raw_cells = door.get("cells") or [[door["x"], door["y"]]]
        doors.append(
            Door(
                id=door["id"],
                cells=[(int(x), int(y)) for x, y in raw_cells],
                opened_by=list(door["opened_by"]),
                logic=door.get("logic", doors_doc.get("logic", "any")),
                mode=door.get("mode", doors_doc.get("mode", "hold")),
                blocks_agent=None if door.get("blocks_agent") is None else str(door["blocks_agent"]),
            )
        )
    objects = []
    for obj in objects_doc.get("objects", []):
        ox, oy = _center(int(obj["x"]), int(obj["y"]))
        objects.append(Crate(obj["id"], ox, oy, obj.get("pushes_onto_plate")))
    goals = []
    for goal in goals_doc["goals"]:
        gx, gy = _center(int(goal["x"]), int(goal["y"]))
        goals.append(Goal(goal["id"], gx, gy, str(goal["agent"])))
    s1 = _center(int(agent1["x"]), int(agent1["y"]))
    s2 = _center(int(agent2["x"]), int(agent2["y"]))
    return MapWorld(
        map_id=str(walls_doc["map_id"]),
        map_type=walls_doc["map_type"],
        width=int(walls_doc["width"]),
        height=int(walls_doc["height"]),
        walls=walls,
        plates=plates,
        doors=doors,
        objects=objects,
        goals=goals,
        starts=[s1, s2],
        independent=bool(walls_doc["independent_regions"]),
    )


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _obs_dim() -> int:
    # 자기 자세 4, 상대 자세 4, 발판/문/물건/목표 각 4슬롯, 통신 4채널.
    per_entity = 4
    comm = N_CHANNELS * (VOCAB + 3)
    return 8 + per_entity * (MAX_PLATES + MAX_DOORS + MAX_OBJECTS + MAX_GOALS) + comm


OBS_DIM = _obs_dim()


class EscapeEnv:
    def __init__(self, maps_dir: str | Path, cfg: TrainConfig | None = None):
        self.cfg = cfg or TrainConfig()
        self.maps_dir = Path(maps_dir)
        self.folders = sorted(path for path in self.maps_dir.iterdir() if path.is_dir())
        if not self.folders:
            raise FileNotFoundError(f"no maps in {self.maps_dir}")
        self.rng = np.random.default_rng(self.cfg.seed)
        self.world: MapWorld | None = None
        self.agents: list[Body] = []
        self.plates: list[Plate] = []
        self.doors: list[Door] = []
        self.objects: list[Crate] = []
        self.goals: list[Goal] = []
        self.walls: set[tuple[int, int]] = set()
        self.board = Blackboard()
        self.step_count = 0
        self.width = 1
        self.height = 1
        self.plate_by_id: dict[str, Plate] = {}

    def reset(self, index: int | None = None) -> np.ndarray:
        if index is None:
            index = int(self.rng.integers(0, len(self.folders)))
        self.world = load_map(self.folders[index])
        self.width = self.world.width
        self.height = self.world.height
        self.walls = set(self.world.walls)
        # 맵을 고칠 때마다 발판 래치를 새로 시작한다.
        self.plates = [
            Plate(plate.id, plate.x, plate.y, list(plate.opens), plate.mode, plate.owner)
            for plate in self.world.plates
        ]
        self.doors = [
            Door(door.id, list(door.cells), list(door.opened_by), door.logic, door.mode, door.blocks_agent)
            for door in self.world.doors
        ]
        self.objects = [Crate(obj.id, obj.x, obj.y, obj.onto_plate) for obj in self.world.objects]
        self.goals = [Goal(goal.id, goal.x, goal.y, goal.agent) for goal in self.world.goals]
        self.plate_by_id = {plate.id: plate for plate in self.plates}
        self.agents = []
        for start, goal in ((self.world.starts[0], self.goal_for(0)), (self.world.starts[1], self.goal_for(1))):
            theta = math.atan2(goal.y - start[1], goal.x - start[0])
            self.agents.append(Body(start[0], start[1], theta))
        self.board.reset()
        self.step_count = 0
        self._refresh_triggers()
        return self.observe()

    def step(self, actions: list[dict]) -> tuple[np.ndarray, float, bool, bool, dict]:
        """actions[i] = {v, omega, send, channel, symbol}. 보상은 팀 공유다."""
        before = potential(self, self.cfg)
        # 통신은 행동 적용 전에 관측된 슬롯을 기준으로 하고, 쓰기는 스텝 끝에 반영된다.
        # 같은 스텝의 송신은 상대 관측에 바로 들어가지 않는다 (한 스텝 지연).
        pending = []
        for index, action in enumerate(actions):
            if int(action["send"]) == 1:
                pending.append((index + 1, int(action["channel"]), int(action["symbol"])))

        self._push_objects(actions)
        for index, action in enumerate(actions):
            self._move_agent(index, float(action["v"]), float(action["omega"]))

        for speaker, channel, symbol in pending:
            self.board.write(speaker, channel, symbol)

        self._refresh_triggers()
        self.step_count += 1
        success = self.success()
        reward = team_reward(self, before, success, self.cfg)
        terminated = success
        truncated = (not success) and self.step_count >= self.cfg.episode_steps
        info = {
            "success": success,
            "map_id": self.world.map_id if self.world else "",
            "map_type": self.world.map_type if self.world else "",
            "channels": [(slot.symbol, slot.speaker) for slot in self.board.slots],
        }
        return self.observe(), float(reward), terminated, truncated, info

    def observe(self) -> np.ndarray:
        return np.stack([self._obs(0), self._obs(1)], axis=0).astype(np.float32)

    def success(self) -> bool:
        if self.world is None:
            return False
        if self.world.independent:
            return all(self.on_trigger(agent.x, agent.y, self.goal_for(index)) for index, agent in enumerate(self.agents))
        goal = self.goals[0]
        return all(self.on_trigger(agent.x, agent.y, goal) for agent in self.agents)

    def goal_for(self, index: int) -> Goal:
        want = "1" if index == 0 else "2"
        for goal in self.goals:
            if goal.agent == want:
                return goal
        return self.goals[0]

    def plate_for(self, index: int) -> Plate | None:
        """분리 맵에서는 그 에이전트가 밟도록 배치된 발판. 없으면 가장 가까운 발판."""
        if not self.plates:
            return None
        want = "1" if index == 0 else "2"
        owned = [plate for plate in self.plates if plate.owner == want]
        if owned:
            return owned[0]
        agent = self.agents[index]
        return min(self.plates, key=lambda plate: self.dist(agent.x, agent.y, plate.x, plate.y))

    def door_blocking(self, index: int) -> Door | None:
        if not self.world or not self.world.independent:
            return next((door for door in self.doors if not door.open), None)
        want = "1" if index == 0 else "2"
        for door in self.doors:
            if door.blocks_agent == want:
                return door
        return None

    def on_trigger(self, x: float, y: float, target) -> bool:
        return self.dist(x, y, target.x, target.y) <= TRIGGER_RADIUS

    @staticmethod
    def dist(x0: float, y0: float, x1: float, y1: float) -> float:
        return math.hypot(x0 - x1, y0 - y1)

    def _obs(self, index: int) -> np.ndarray:
        agent = self.agents[index]
        other = self.agents[1 - index]
        parts = [
            self._pose(agent),
            np.array(
                [
                    (other.x - agent.x) / self.width,
                    (other.y - agent.y) / self.height,
                    math.cos(other.theta),
                    math.sin(other.theta),
                ],
                dtype=np.float32,
            ),
        ]
        parts.append(self._pad_points(
            [(plate.x, plate.y, 1.0 if plate.active else 0.0) for plate in self.plates],
            MAX_PLATES,
            agent,
        ))
        parts.append(self._pad_points(
            [(self._door_anchor(door), 1.0 if door.open else 0.0) for door in self.doors],
            MAX_DOORS,
            agent,
            unpack_xy=True,
        ))
        attached = self._attached_index()
        parts.append(self._pad_points(
            [(obj.x, obj.y, 1.0 if index in attached[obj_i] else 0.0) for obj_i, obj in enumerate(self.objects)],
            MAX_OBJECTS,
            agent,
        ))
        goals = [self.goal_for(index), self.goal_for(1 - index)]
        # 같은 목표를 둘이 쓰면 두 번째 슬롯은 가린다.
        if goals[1] is goals[0]:
            goal_rows = [(goals[0].x, goals[0].y, 1.0 if self.on_trigger(agent.x, agent.y, goals[0]) else 0.0)]
        else:
            goal_rows = [
                (goal.x, goal.y, 1.0 if self.on_trigger(agent.x, agent.y, goal) else 0.0)
                for goal in goals
            ]
        parts.append(self._pad_points(goal_rows, MAX_GOALS, agent))
        parts.append(self.board.features())
        obs = np.concatenate(parts)
        if obs.shape[0] != OBS_DIM:
            raise RuntimeError(f"obs dim {obs.shape[0]} != {OBS_DIM}")
        return obs

    def _pose(self, agent: Body) -> np.ndarray:
        return np.array(
            [agent.x / self.width, agent.y / self.height, math.cos(agent.theta), math.sin(agent.theta)],
            dtype=np.float32,
        )

    def _pad_points(self, rows, limit: int, agent: Body, unpack_xy: bool = False) -> np.ndarray:
        packed = []
        for row in rows[:limit]:
            if unpack_xy:
                (x, y), flag = row
            else:
                x, y, flag = row
            packed.extend([(x - agent.x) / self.width, (y - agent.y) / self.height, flag, 1.0])
        while len(packed) < limit * 4:
            packed.extend([0.0, 0.0, 0.0, 0.0])
        return np.array(packed, dtype=np.float32)

    def _door_anchor(self, door: Door) -> tuple[float, float]:
        xs = [cell[0] + 0.5 for cell in door.cells]
        ys = [cell[1] + 0.5 for cell in door.cells]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _attached_index(self) -> list[set[int]]:
        """물건마다, 그 물건에 붙어 있는 에이전트 번호."""
        reach = AGENT_RADIUS + OBJECT_RADIUS + ATTACH_GAP
        attached = []
        for obj in self.objects:
            who = set()
            for index, agent in enumerate(self.agents):
                if self.dist(agent.x, agent.y, obj.x, obj.y) <= reach:
                    who.add(index)
            attached.append(who)
        return attached

    def _push_objects(self, actions: list[dict]) -> None:
        attached = self._attached_index()
        for obj_index, obj in enumerate(self.objects):
            who = attached[obj_index]
            if who != {0, 1}:
                continue
            speeds = [float(actions[index]["v"]) for index in (0, 1)]
            if min(speeds) < PUSH_SPEED:
                continue
            headings = [
                (math.cos(self.agents[index].theta), math.sin(self.agents[index].theta))
                for index in (0, 1)
            ]
            push_x = headings[0][0] + headings[1][0]
            push_y = headings[0][1] + headings[1][1]
            norm = math.hypot(push_x, push_y)
            # 서로 반대 방향을 보면 합이 거의 0이라 물건은 움직이지 않는다.
            if norm < 0.5:
                continue
            distance = min(speeds) * DT
            nx, ny = obj.x + distance * push_x / norm, obj.y + distance * push_y / norm
            if not self._circle_hits_static(nx, ny, OBJECT_RADIUS):
                obj.x, obj.y = nx, ny

    def _move_agent(self, index: int, speed: float, omega: float) -> None:
        agent = self.agents[index]
        agent.theta = (agent.theta + omega * DT + math.pi) % (2 * math.pi) - math.pi
        dx = speed * math.cos(agent.theta) * DT
        dy = speed * math.sin(agent.theta) * DT
        agent.x, agent.y = self._slide(agent.x, agent.y, dx, dy, AGENT_RADIUS, ignore_agents=True)

    def _slide(self, x: float, y: float, dx: float, dy: float, radius: float, ignore_agents: bool) -> tuple[float, float]:
        if not self._blocked(x + dx, y + dy, radius):
            return x + dx, y + dy
        if not self._blocked(x + dx, y, radius):
            return x + dx, y
        if not self._blocked(x, y + dy, radius):
            return x, y + dy
        return x, y

    def _blocked(self, x: float, y: float, radius: float) -> bool:
        if self._circle_hits_static(x, y, radius):
            return True
        for obj in self.objects:
            if self.dist(x, y, obj.x, obj.y) < radius + OBJECT_RADIUS:
                return True
        return False

    def _circle_hits_static(self, x: float, y: float, radius: float) -> bool:
        blocked = set(self.walls)
        for door in self.doors:
            if not door.open:
                blocked.update(door.cells)
        x0 = int(math.floor(x - radius))
        x1 = int(math.floor(x + radius))
        y0 = int(math.floor(y - radius))
        y1 = int(math.floor(y + radius))
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                if (cx, cy) not in blocked:
                    continue
                nearest_x = min(max(x, cx), cx + 1.0)
                nearest_y = min(max(y, cy), cy + 1.0)
                if (x - nearest_x) ** 2 + (y - nearest_y) ** 2 < radius ** 2:
                    return True
        return False

    def _refresh_triggers(self) -> None:
        for plate in self.plates:
            occupied = any(self.on_trigger(agent.x, agent.y, plate) for agent in self.agents)
            occupied = occupied or any(self.on_trigger(obj.x, obj.y, plate) for obj in self.objects)
            plate.occupied = occupied
            if occupied and plate.mode == "latch":
                plate.latched = True
        by_id = self.plate_by_id
        for door in self.doors:
            flags = [by_id[plate_id].active for plate_id in door.opened_by if plate_id in by_id]
            if not flags:
                door.open = False
                continue
            if door.logic == "all":
                door.open = all(flags)
            else:
                door.open = any(flags)


def run_mechanics_checks(maps_dir: str | Path) -> None:
    """겹침, 혼자 밀기 금지, 채널 저장이 스펙대로인지 확인한다."""
    env = EscapeEnv(maps_dir, TrainConfig(episode_steps=20))
    chosen = None
    for path in env.folders:
        doc = _read(path / "objects.yaml")
        if doc.get("objects"):
            chosen = env.folders.index(path)
            break
    if chosen is None:
        raise RuntimeError("no map with an object")
    env.reset(chosen)
    obj = env.objects[0]
    start = (obj.x, obj.y)
    # 한 명만 붙어서 민다.
    env.agents[0].x, env.agents[0].y = obj.x - 0.6, obj.y
    env.agents[0].theta = 0.0
    env.agents[1].x, env.agents[1].y = obj.x, obj.y + 5.0
    env.agents[1].theta = 0.0
    idle = {"v": 0.0, "omega": 0.0, "send": 0, "channel": 0, "symbol": 0}
    push = {"v": 2.0, "omega": 0.0, "send": 0, "channel": 0, "symbol": 0}
    env.step([push, idle])
    if abs(env.objects[0].x - start[0]) > 1e-6 or abs(env.objects[0].y - start[1]) > 1e-6:
        raise RuntimeError("a single agent moved the object")

    # 둘 다 같은 좌표에 있을 수 있다.
    env.agents[0].x = env.agents[1].x = obj.x - 0.55
    env.agents[0].y = env.agents[1].y = obj.y
    env.agents[0].theta = env.agents[1].theta = 0.0
    before = (env.objects[0].x, env.objects[0].y)
    env.step([push, push])
    if (env.objects[0].x, env.objects[0].y) == before and not env._circle_hits_static(before[0] + 0.2, before[1], OBJECT_RADIUS):
        raise RuntimeError("two attached agents did not move a free object")
    if abs(env.agents[0].x - env.agents[1].x) > 1.5:
        raise RuntimeError("agents were separated even though overlap is allowed")

    # 채널에 쓴 기호는 다음 관측까지 남는다.
    env.step([
        {"v": 0.0, "omega": 0.0, "send": 1, "channel": 2, "symbol": 5},
        idle,
    ])
    obs = env.observe()
    # 통신 특징은 관측 맨 끝. 채널 2의 기호 원핫이 5인지 본다.
    comm = obs[1, -N_CHANNELS * (VOCAB + 3):]
    slot = comm[2 * (VOCAB + 3): 3 * (VOCAB + 3)]
    if int(np.argmax(slot[:VOCAB])) != 5:
        raise RuntimeError("channel 2 did not keep the written symbol")
    print("mechanics checks passed", {"map": env.world.map_id, "object_shift": round(env.objects[0].x - before[0], 3)})
