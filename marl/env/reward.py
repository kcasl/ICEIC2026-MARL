"""팀 보상.

희소 성공 보상과 시간 벌점에, Ng, Harada, Russell 1999의 잠재함수 셰이핑을 더한다.
Φ는 상태만의 함수이고 F = γΦ(s') - Φ(s) 라서, 계수를 곱해도 최적 정책은
희소 보상만 쓸 때와 같다. 이벤트(발판, 문)와 목표까지 남은 거리를 반반 섞는다.
일회성 발판 보너스는 넣지 않는다. 잠재함수와 겹치면 정책이 기울어진다.
"""

from __future__ import annotations

import math

from marl.config import TrainConfig


def _closeness(distance: float, scale: float) -> float:
    if scale <= 1e-6:
        return 1.0
    return max(0.0, 1.0 - distance / scale)


def potential(env, cfg: TrainConfig) -> float:
    """0에 가깝면 아직 아무 진전도 없고, 1에 가까우면 탈출이 끝난 상태다."""
    scale = math.hypot(env.width, env.height)
    plate_frac = _mean_flag(env.plates, "active")
    door_frac = _mean_flag(env.doors, "open")

    goal_hits = []
    progress = []
    for index, agent in enumerate(env.agents):
        goal = env.goal_for(index)
        arrived = env.on_trigger(agent.x, agent.y, goal)
        goal_hits.append(1.0 if arrived else 0.0)
        # 문이 아직 닫혀 있으면 목표 거리 대신, 그 에이전트가 밟을 발판까지의 거리.
        if env.door_blocking(index) is not None and not env.door_blocking(index).open:
            plate = env.plate_for(index)
            target = plate if plate is not None else goal
            progress.append(_closeness(env.dist(agent.x, agent.y, target.x, target.y), scale))
        else:
            progress.append(_closeness(env.dist(agent.x, agent.y, goal.x, goal.y), scale))

    object_terms = []
    for obj in env.objects:
        plate = env.plate_by_id.get(obj.onto_plate) if obj.onto_plate else None
        if plate is None:
            continue
        if env.on_trigger(obj.x, obj.y, plate):
            object_terms.append(1.0)
        else:
            object_terms.append(_closeness(env.dist(obj.x, obj.y, plate.x, plate.y), scale))

    event = 0.5 * plate_frac + 0.5 * door_frac
    move = sum(progress) / len(progress)
    if object_terms:
        move = 0.5 * move + 0.5 * (sum(object_terms) / len(object_terms))
    goal_frac = sum(goal_hits) / len(goal_hits)
    return 0.5 * event + 0.5 * (0.5 * goal_frac + 0.5 * move)


def team_reward(env, before: float, success: bool, cfg: TrainConfig) -> float:
    after = potential(env, cfg)
    shaping = cfg.gamma * after - before
    bonus = cfg.success_bonus if success else 0.0
    return bonus - cfg.time_penalty + cfg.shaping_coef * shaping


def _mean_flag(items, name: str) -> float:
    if not items:
        return 1.0
    return sum(1.0 if getattr(item, name) else 0.0 for item in items) / len(items)
