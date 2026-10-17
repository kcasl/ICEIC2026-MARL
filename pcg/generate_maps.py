"""Tile assets and procedural maps for cooperative escape.

Walls are black and floors are plain white. Map types:

- crate_straight / crate_bend / crate_s / crate_chain
  One plate before each door. A crate starts beside that plate so an agent
  can push it onto the plate and walk through while the plate stays held.
- split_vertical / split_horizontal
  Two regions with no path between them. Each door has a plate on its
  approach side. Both plates must be pressed before either door stays open.
- paired_offset / paired_l
  Connected rooms with a plate on both sides of the door.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"
TILE_DIR = ASSET_DIR / "tiles"
CELL = 32

FLOOR = "floor"
WALL = "wall"
PLATE = "plate"
DOOR = "door"
OBJECT = "object"
GOAL = "goal"
AGENT1 = "agent1_start"
AGENT2 = "agent2_start"
AGENT1_UNIT = "agent1"
AGENT2_UNIT = "agent2"

SAMPLE = (2, 2)

TILES = {
    WALL: {"name": "벽", "color": (0, 0, 0), "blocks": True},
    FLOOR: {"name": "바닥", "color": (255, 255, 255), "blocks": False},
    PLATE: {"name": "발판", "color": (245, 196, 0), "blocks": False},
    DOOR: {"name": "문", "color": (176, 58, 46), "blocks": False},
    OBJECT: {"name": "옮길 수 있는 오브젝트", "color": (146, 98, 56), "blocks": False},
    GOAL: {"name": "도착지점", "color": (30, 160, 90), "blocks": False},
    AGENT1: {"name": "Agent1 시작지점", "color": (36, 120, 190), "blocks": False},
    AGENT2: {"name": "Agent2 시작지점", "color": (150, 70, 180), "blocks": False},
    AGENT1_UNIT: {"name": "Agent1", "color": (36, 120, 190), "blocks": False},
    AGENT2_UNIT: {"name": "Agent2", "color": (150, 70, 180), "blocks": False},
}

DIRS = {
    "east": (1, 0),
    "west": (-1, 0),
    "south": (0, 1),
    "north": (0, -1),
}

# Two empty cells sit between a crate and the plate it is pushed onto.
MIN_PLATE_OBJECT_DISTANCE = 3

TYPE_NOTES = {
    "crate_straight": "복도. 문 앞 발판이 하나이고, 물건을 발판 위로 밀어 문을 연 뒤 지나간다.",
    "crate_bend": "ㄱ자 복도. 문 앞 발판이 하나이고, 물건을 발판에 올린 뒤 지나간다.",
    "crate_s": "S자 복도. 문 앞 발판이 하나이고, 물건을 발판에 올린 뒤 지나간다.",
    "crate_chain": "문이 두 개. 각 문 앞에 발판과 물건이 있어 순서대로 올리고 지나간다.",
    "split_vertical": "좌우로 완전히 분리. 문마다 들어가는 쪽에 발판이 있고, 두 발판을 모두 밟아야 문이 열린다.",
    "split_horizontal": "상하로 완전히 분리. 문마다 들어가는 쪽에 발판이 있고, 두 발판을 모두 밟아야 문이 열린다.",
    "paired_offset": "어긋난 두 방. 문 양쪽에 발판이 있다.",
    "paired_l": "ㄴ자 시작 구역과 옆방. 문 양쪽에 발판이 있다.",
}


@dataclass
class Plate:
    id: str
    cell: tuple[int, int]
    opens: list[str]
    region: int


@dataclass
class Door:
    id: str
    cell: tuple[int, int]
    opened_by: list[str]
    # 문 개구는 한 칸이 아니라 이 칸들을 포함해 세 칸이다. 두 에이전트가 나란히 지나간다.
    span: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Crate:
    id: str
    cell: tuple[int, int]
    pushes: list[dict]
    stand_cells: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Goal:
    id: str
    cell: tuple[int, int]
    agent: str
    side: str


@dataclass
class Spec:
    map_id: str
    seed: int
    map_type: str
    width: int
    height: int
    axis: str
    start_side: str
    goal_side: str
    grid: list[list[str]]
    plates: list[Plate]
    doors: list[Door]
    crates: list[Crate]
    goals: list[Goal]
    agent1: tuple[int, int]
    agent2: tuple[int, int]
    separated: bool
    independent: bool
    clear_condition: str
    regions: dict = field(default_factory=dict)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def draw_tiles() -> None:
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    for key, spec in TILES.items():
        image = Image.new("RGB", (CELL, CELL), spec["color"])
        draw = ImageDraw.Draw(image)
        if key == PLATE:
            draw.rounded_rectangle((6, 6, 25, 25), radius=4, fill=(196, 145, 0))
            draw.rectangle((13, 13, 18, 18), fill=(255, 220, 80))
        elif key == DOOR:
            draw.rectangle((6, 3, 25, 28), outline=(96, 28, 22), width=2)
            draw.line((16, 5, 16, 26), fill=(96, 28, 22), width=2)
            draw.ellipse((19, 14, 23, 18), fill=(245, 196, 0))
        elif key == OBJECT:
            draw.rectangle((5, 5, 26, 26), outline=(92, 58, 28), width=2)
            draw.line((7, 7, 24, 24), fill=(92, 58, 28), width=2)
            draw.line((24, 7, 7, 24), fill=(92, 58, 28), width=2)
        elif key == GOAL:
            draw.ellipse((4, 4, 27, 27), fill=(18, 120, 68))
            draw.ellipse((11, 11, 20, 20), fill=(210, 255, 220))
        elif key in (AGENT1, AGENT2):
            label = "1" if key == AGENT1 else "2"
            draw.ellipse((4, 4, 27, 27), fill=(255, 255, 255))
            draw.text((12, 9), label, fill=spec["color"])
        elif key in (AGENT1_UNIT, AGENT2_UNIT):
            draw.ellipse((11, 5, 20, 14), fill=(255, 255, 255))
            draw.rounded_rectangle((9, 15, 22, 27), radius=4, fill=(255, 255, 255))
            draw.ellipse((13, 8, 15, 10), fill=spec["color"])
            draw.ellipse((16, 8, 18, 10), fill=spec["color"])
        path = TILE_DIR / f"{key}.png"
        image.save(path)
        if image.getpixel(SAMPLE) != spec["color"]:
            raise RuntimeError(f"{key} sample pixel is not the feature color")
        write_yaml(
            ASSET_DIR / f"{key}.yaml",
            {
                "id": key,
                "name": spec["name"],
                "file": f"assets/tiles/{key}.png",
                "cell_size": CELL,
                "color": list(spec["color"]),
                "sample_pixel": list(SAMPLE),
                "blocks_movement": spec["blocks"],
                **({"kind": "agent"} if key in (AGENT1_UNIT, AGENT2_UNIT) else {}),
            },
        )


def load_tiles() -> dict[str, Image.Image]:
    return {key: Image.open(TILE_DIR / f"{key}.png").convert("RGB") for key in TILES}


def blank(width: int, height: int) -> list[list[str]]:
    return [[WALL for _ in range(width)] for _ in range(height)]


def inside(grid: list[list[str]], x: int, y: int) -> bool:
    return 0 < x < len(grid[0]) - 1 and 0 < y < len(grid) - 1


def carve_rect(grid: list[list[str]], x0: int, y0: int, x1: int, y1: int) -> None:
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if inside(grid, x, y):
                grid[y][x] = FLOOR


def carve(grid: list[list[str]], cells: list[tuple[int, int]]) -> None:
    for x, y in cells:
        if not inside(grid, x, y):
            raise ValueError(f"cell {(x, y)} is outside the map")
        grid[y][x] = FLOOR


def direction_name(dx: int, dy: int) -> str:
    for name, step in DIRS.items():
        if step == (dx, dy):
            return name
    raise ValueError(f"not a cardinal step: {(dx, dy)}")


def push_record(origin: tuple[int, int], target: tuple[int, int], onto_plate: str | None) -> dict:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    steps = max(abs(dx), abs(dy))
    if steps == 0 or abs(dx) + abs(dy) != steps:
        raise ValueError(f"push from {origin} to {target} is not a straight line")
    return {
        "direction": direction_name(dx // steps, dy // steps),
        "x": target[0],
        "y": target[1],
        "onto_plate": onto_plate,
        "distance": steps,
        "requires_both_agents": True,
    }


def carve_coop_lane(
    grid: list[list[str]],
    behind: tuple[int, int],
    assist: tuple[int, int],
    direction: tuple[int, int],
    distance: int = MIN_PLATE_OBJECT_DISTANCE,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Carve a 2-agent push pocket and a straight lane onto a plate.

    `behind` is the cell the pushing agent occupies. `assist` is beside that
    cell so the second agent can stand there at the same time. The crate sits
    one step ahead of `behind`, then `distance` steps further is the plate.
    """
    dx, dy = direction
    if abs(assist[0] - behind[0]) + abs(assist[1] - behind[1]) != 1:
        return None
    crate = (behind[0] + dx, behind[1] + dy)
    plate = (crate[0] + distance * dx, crate[1] + distance * dy)
    cells = [behind, assist, crate, plate]
    cells.extend((crate[0] + i * dx, crate[1] + i * dy) for i in range(1, distance))
    if len(set(cells)) != len(cells) or any(not inside(grid, *cell) for cell in cells):
        return None
    # 밀기 통로도 세 칸 폭으로 깐다. 한 칸짜리는 둘이 붙어서 밀 자리가 없다.
    lateral = (assist[0] - behind[0], assist[1] - behind[1])
    wide: list[tuple[int, int]] = []
    for cell in cells:
        for sign in (1, -1):
            wide.append((cell[0] + sign * lateral[0], cell[1] + sign * lateral[1]))
    if any(not inside(grid, *cell) for cell in wide):
        return None
    carve(grid, cells + wide)
    return crate, plate


def paint_door(
    grid: list[list[str]],
    anchor: tuple[int, int],
    lateral: tuple[int, int],
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    """문을 세 칸 폭으로 연다. 가운데 칸이 대표 좌표다."""
    lx, ly = lateral
    cells = [
        (anchor[0] - lx, anchor[1] - ly),
        anchor,
        (anchor[0] + lx, anchor[1] + ly),
    ]
    if any(not inside(grid, *cell) for cell in cells):
        return None
    for x, y in cells:
        grid[y][x] = DOOR
    return cells[1], [cells[0], cells[2]]


def flood(origin: tuple[int, int], passable: set[tuple[int, int]]) -> set[tuple[int, int]]:
    if origin not in passable:
        return set()
    seen = {origin}
    stack = [origin]
    while stack:
        x, y = stack.pop()
        for dx, dy in DIRS.values():
            nxt = (x + dx, y + dy)
            if nxt in passable and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def passable_cells(
    grid: list[list[str]],
    blocked: set[tuple[int, int]],
    open_doors: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    cells = set()
    for y, row in enumerate(grid):
        for x, kind in enumerate(row):
            cell = (x, y)
            if cell in blocked or kind == WALL:
                continue
            if kind == DOOR and cell not in open_doors:
                continue
            cells.add(cell)
    return cells


def door_cells(spec: Spec) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for door in spec.doors:
        cells.update(cells_of(door))
    return cells


def cells_of(door: Door) -> set[tuple[int, int]]:
    return {door.cell, *door.span}


def validate_clearance(spec: Spec) -> None:
    """한 칸짜리 목은 거른다. 두 에이전트가 나란히 설 폭이 남아 있어야 한다."""
    height, width = spec.height, spec.width

    def is_wall(x: int, y: int) -> bool:
        if not (0 <= x < width and 0 <= y < height):
            return True
        return spec.grid[y][x] == WALL

    for y in range(height):
        for x in range(width):
            if spec.grid[y][x] == WALL:
                continue
            if (is_wall(x - 1, y) and is_wall(x + 1, y)) or (is_wall(x, y - 1) and is_wall(x, y + 1)):
                raise AssertionError(f"passage at {(x, y)} is only one cell wide")


def opposite(side: str) -> str:
    return {"west": "east", "east": "west", "north": "south", "south": "north"}[side]


def axis_size(spec: Spec) -> int:
    return spec.width if spec.axis == "x" else spec.height


def coord(cell: tuple[int, int], axis: str) -> int:
    return cell[0] if axis == "x" else cell[1]


def faces(spec: Spec, starts: list[tuple[int, int]], goals: list[tuple[int, int]]) -> None:
    center = (axis_size(spec) - 1) / 2
    for start in starts:
        for goal in goals:
            delta = (coord(start, spec.axis) - center) * (coord(goal, spec.axis) - center)
            if delta >= 0:
                raise AssertionError("start and goal are not on opposite sides")
    if spec.goal_side != opposite(spec.start_side):
        raise AssertionError("goal side label is not opposite the start side")


def crate_for(spec: Spec, plate_id: str) -> Crate | None:
    for crate in spec.crates:
        if any(push.get("onto_plate") == plate_id for push in crate.pushes):
            return crate
    return None


def plate_target(crate: Crate, plate_id: str) -> tuple[int, int]:
    for push in crate.pushes:
        if push.get("onto_plate") == plate_id:
            return (push["x"], push["y"])
    raise AssertionError(f"{crate.id} does not push onto {plate_id}")


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def line_step(origin: tuple[int, int], target: tuple[int, int]) -> tuple[int, int, int]:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    steps = max(abs(dx), abs(dy))
    if steps == 0 or abs(dx) + abs(dy) != steps:
        raise AssertionError("push lane is not a straight cardinal line")
    return dx // steps, dy // steps, steps


def behind_cell(crate: Crate, plate_id: str) -> tuple[int, int]:
    step_x, step_y, _ = line_step(crate.cell, plate_target(crate, plate_id))
    return (crate.cell[0] - step_x, crate.cell[1] - step_y)


def validate_spacing(spec: Spec) -> None:
    for crate in spec.crates:
        for plate in spec.plates:
            if manhattan(crate.cell, plate.cell) < MIN_PLATE_OBJECT_DISTANCE:
                raise AssertionError(f"{crate.id} is within {MIN_PLATE_OBJECT_DISTANCE - 1} cells of {plate.id}")
        if len(crate.stand_cells) != 2 or manhattan(crate.stand_cells[0], crate.stand_cells[1]) != 1:
            raise AssertionError(f"{crate.id} has no side-by-side stand for two agents")
        if all(manhattan(stand, crate.cell) != 1 for stand in crate.stand_cells):
            raise AssertionError(f"{crate.id} cannot be pushed by an agent standing beside it")
        if not spec.map_type.startswith("crate"):
            origins = [spec.agent1, spec.agent2]
            if spec.independent:
                own = reaches(spec, spec.agent1, set(), set())
                origins = [spec.agent1] if crate.cell in own else [spec.agent2]
            for origin in origins:
                seen = reaches(spec, origin, {crate.cell}, set())
                if any(stand not in seen for stand in crate.stand_cells):
                    raise AssertionError(f"an agent cannot reach both stand cells of {crate.id}")
        for push in crate.pushes:
            if not push.get("onto_plate"):
                continue
            step_x, step_y, steps = line_step(crate.cell, (push["x"], push["y"]))
            if steps < MIN_PLATE_OBJECT_DISTANCE:
                raise AssertionError(f"{crate.id} push onto a plate is shorter than {MIN_PLATE_OBJECT_DISTANCE}")
            if behind_cell(crate, push["onto_plate"]) not in crate.stand_cells:
                raise AssertionError(f"{crate.id} stand is not behind the push")
            for index in range(1, steps):
                cell = (crate.cell[0] + step_x * index, crate.cell[1] + step_y * index)
                if spec.grid[cell[1]][cell[0]] != FLOOR:
                    raise AssertionError(f"{crate.id} push lane is blocked")


def reaches(spec: Spec, origin: tuple[int, int], blocked: set[tuple[int, int]], open_doors: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return flood(origin, passable_cells(spec.grid, blocked, open_doors))


def validate(spec: Spec) -> None:
    validate_spacing(spec)
    validate_clearance(spec)
    validate_plates_for_doors(spec)
    if spec.map_type.startswith("crate"):
        validate_crate(spec)
    elif spec.map_type.startswith("split"):
        validate_split(spec)
    elif spec.map_type.startswith("paired"):
        validate_paired(spec)
    else:
        raise AssertionError(f"unknown map type {spec.map_type}")
    claimed: dict[tuple[int, int], str] = {}
    for plate in spec.plates:
        claimed[plate.cell] = plate.id
    for door in spec.doors:
        for cell in (door.cell, *door.span):
            if cell in claimed and claimed[cell] != door.id:
                raise AssertionError("overlapping door")
            claimed[cell] = door.id
    for crate in spec.crates:
        if crate.cell in claimed:
            raise AssertionError("overlapping crate")
        claimed[crate.cell] = crate.id
    for goal in spec.goals:
        if goal.cell in claimed:
            raise AssertionError("overlapping goal")
        claimed[goal.cell] = goal.id
    for agent in (spec.agent1, spec.agent2):
        if agent in claimed:
            raise AssertionError("agent overlaps a feature")


def validate_crate(spec: Spec) -> None:
    faces(spec, [spec.agent1, spec.agent2], [goal.cell for goal in spec.goals])
    if spec.independent or len(spec.goals) != 1:
        raise AssertionError("crate map should share one goal")
    ordered = sorted(spec.doors, key=lambda door: min(abs(door.cell[0] - spec.agent1[0]) + abs(door.cell[1] - spec.agent1[1]), abs(door.cell[0] - spec.agent2[0]) + abs(door.cell[1] - spec.agent2[1])))
    # Order gates by graph distance from the starts with every door open.
    opened = door_cells(spec)
    distances = []
    region = reaches(spec, spec.agent1, set(), opened)
    if spec.agent2 not in region:
        raise AssertionError("agents cannot meet")
    for door in spec.doors:
        if door.cell not in region:
            raise AssertionError("door is not on the shared path")
        distances.append((abs(door.cell[0] - spec.agent1[0]) + abs(door.cell[1] - spec.agent1[1]), door))
    # BFS order is more reliable than manhattan when the path bends.
    ordered = []
    remaining = {door.id: door for door in spec.doors}
    frontier_blocked: set[tuple[int, int]] = set()
    while remaining:
        seen = reaches(spec, spec.agent1, frontier_blocked, set().union(*(cells_of(door) for door in spec.doors if door.id not in remaining)))
        # doors still closed: find one whose entry is already reached
        ready = []
        for door in remaining.values():
            entries = [
                (door.cell[0] + dx, door.cell[1] + dy)
                for dx, dy in DIRS.values()
                if (door.cell[0] + dx, door.cell[1] + dy) in seen
            ]
            if entries:
                ready.append(door)
        if len(ready) != 1 and not (len(ready) == 1 or (not ordered and len(ready) >= 1)):
            if len(ready) < 1:
                raise AssertionError("no reachable closed door")
        door = min(ready, key=lambda item: abs(item.cell[0] - spec.agent1[0]) + abs(item.cell[1] - spec.agent1[1]))
        ordered.append(door)
        del remaining[door.id]

    lifted: set[tuple[int, int]] = set()
    parked: set[tuple[int, int]] = set()
    open_now: set[tuple[int, int]] = set()
    for door in ordered:
        if len(door.opened_by) != 1:
            raise AssertionError("a crate door must have exactly one plate")
        plate_id = door.opened_by[0]
        plate = next(item for item in spec.plates if item.id == plate_id)
        crate = crate_for(spec, plate_id)
        if crate is None:
            raise AssertionError("plate before the door has no crate")
        if plate_target(crate, plate_id) != plate.cell:
            raise AssertionError("crate does not push onto the plate cell")
        blocked = {item.cell for item in spec.crates if item.cell not in lifted} | parked
        seen = reaches(spec, spec.agent1, blocked, open_now)
        if spec.agent2 not in seen or any(cell not in seen for cell in crate.stand_cells):
            raise AssertionError("both agents cannot reach the cells used to push the crate")
        if any(goal.cell in seen for goal in spec.goals):
            raise AssertionError("goal is reachable before this door is held open")
        lifted.add(crate.cell)
        parked.add(plate.cell)
        open_now.update(cells_of(door))
    final = reaches(spec, spec.agent1, parked, open_now)
    if spec.agent2 not in final or any(goal.cell not in final for goal in spec.goals):
        raise AssertionError("pushing every crate onto its plate does not open a path to the goal")


def split_roles(spec: Spec) -> tuple[dict[str, str], dict[str, str]]:
    """Return which agent each door blocks, and which agent can press each plate."""
    side1 = reaches(spec, spec.agent1, set(), set())
    side2 = reaches(spec, spec.agent2, set(), set())

    def owner(cell: tuple[int, int]) -> str | None:
        if cell in side1:
            return "1"
        if cell in side2:
            return "2"
        return None

    blocks: dict[str, str] = {}
    for door in spec.doors:
        owners = {
            owner((door.cell[0] + dx, door.cell[1] + dy))
            for dx, dy in DIRS.values()
        }
        owners.discard(None)
        if len(owners) != 1:
            raise AssertionError(f"{door.id} is not sealed into one region")
        blocks[door.id] = owners.pop()
    pressed_by: dict[str, str] = {}
    for plate in spec.plates:
        who = owner(plate.cell)
        if who is None:
            raise AssertionError(f"{plate.id} cannot be pressed while the doors are closed")
        pressed_by[plate.id] = who
    return blocks, pressed_by


def doors_open_for(spec: Spec, pressed_ids: set[str]) -> set[tuple[int, int]]:
    """Split doors open only after every linked plate has been pressed."""
    cells = set()
    for door in spec.doors:
        needed = set(door.opened_by)
        if not needed:
            continue
        if spec.independent:
            if needed <= pressed_ids:
                cells.update(cells_of(door))
        elif needed & pressed_ids:
            cells.update(cells_of(door))
    return cells


def validate_plates_for_doors(spec: Spec) -> None:
    if len(spec.plates) < len(spec.doors):
        raise AssertionError("a map needs at least one plate per door")
    for door in spec.doors:
        if not door.opened_by:
            raise AssertionError(f"{door.id} has no plate")
        others = {item.cell for item in spec.doors if item.id != door.id}
        reachable: set[tuple[int, int]] = set()
        for origin in (spec.agent1, spec.agent2):
            reachable |= reaches(spec, origin, set(), others)
        linked = [plate for plate in spec.plates if plate.id in door.opened_by]
        if not any(plate.cell in reachable for plate in linked):
            raise AssertionError(f"{door.id} has no plate on the approach side")


def validate_split(spec: Spec) -> None:
    if not spec.independent or len(spec.goals) != 2 or len(spec.doors) != 2 or len(spec.plates) != 2:
        raise AssertionError("split map should have two sealed regions")
    if spec.crates:
        raise AssertionError("a sealed region cannot hold a crate that needs both agents")
    goals = {goal.agent: goal for goal in spec.goals}
    faces(spec, [spec.agent1], [goals["1"].cell])
    faces(spec, [spec.agent2], [goals["2"].cell])
    blocks, pressed_by = split_roles(spec)
    plate_ids = {plate.id for plate in spec.plates}
    for door in spec.doors:
        if set(door.opened_by) != plate_ids:
            raise AssertionError(f"{door.id} must require a plate from each region")
        local = [plate for plate in spec.plates if pressed_by[plate.id] == blocks[door.id]]
        if len(local) != 1:
            raise AssertionError(f"{door.id} has no plate in the blocked agent's room")
    all_doors = door_cells(spec)
    if spec.agent2 in reaches(spec, spec.agent1, set(), all_doors):
        raise AssertionError("agents can meet; the regions are not independent")
    agents = {"1": spec.agent1, "2": spec.agent2}
    for key in ("1", "2"):
        own_ids = {plate.id for plate in spec.plates if pressed_by[plate.id] == key}
        alone = reaches(spec, agents[key], set(), doors_open_for(spec, own_ids))
        if goals[key].cell in alone:
            raise AssertionError(f"agent {key} reaches the goal without the other plate")
    solved_doors = doors_open_for(spec, plate_ids)
    if solved_doors != all_doors:
        raise AssertionError("pressing every plate does not open every door")
    solved = reaches(spec, spec.agent1, set(), solved_doors)
    if goals["1"].cell not in solved or spec.agent2 in solved:
        raise AssertionError("both plates do not clear agent 1's side")
    solved2 = reaches(spec, spec.agent2, set(), solved_doors)
    if goals["2"].cell not in solved2 or spec.agent1 in solved2:
        raise AssertionError("both plates do not clear agent 2's side")


def validate_paired(spec: Spec) -> None:
    faces(spec, [spec.agent1, spec.agent2], [goal.cell for goal in spec.goals])
    if len(spec.doors) != 1 or len(spec.plates) != 2 or len(spec.goals) != 1:
        raise AssertionError("paired map needs one door and two plates")
    door = spec.doors[0]
    if set(door.opened_by) != {plate.id for plate in spec.plates}:
        raise AssertionError("both plates should open the shared door")
    closed = reaches(spec, spec.agent1, set(), set())
    if spec.agent2 not in closed:
        raise AssertionError("agents start disconnected")
    near = [plate for plate in spec.plates if plate.cell in closed]
    far = [plate for plate in spec.plates if plate.cell not in closed]
    if len(near) != 1 or len(far) != 1:
        raise AssertionError("the two plates are not on opposite sides of the door")
    if spec.goals[0].cell in closed:
        raise AssertionError("goal is on the start side of the door")
    blocked = {crate.cell for crate in spec.crates}
    opened = reaches(spec, spec.agent1, blocked, cells_of(door))
    if spec.agent2 not in opened or spec.goals[0].cell not in opened or far[0].cell not in opened:
        raise AssertionError("opening the door does not connect both plates and the goal")


def transpose(spec: Spec) -> Spec:
    height = spec.width
    width = spec.height
    grid = [[spec.grid[y][x] for y in range(spec.height)] for x in range(spec.width)]
    # grid[new_y][new_x] = old[new_x][new_y], and new_y = old x, new_x = old y.
    grid = [[FLOOR for _ in range(spec.height)] for _ in range(spec.width)]
    for y in range(spec.height):
        for x in range(spec.width):
            grid[x][y] = spec.grid[y][x]

    def point(cell: tuple[int, int]) -> tuple[int, int]:
        return (cell[1], cell[0])

    side_map = {"west": "north", "east": "south", "north": "west", "south": "east"}
    plates = [Plate(plate.id, point(plate.cell), list(plate.opens), plate.region) for plate in spec.plates]
    doors = [
        Door(door.id, point(door.cell), list(door.opened_by), [point(cell) for cell in door.span])
        for door in spec.doors
    ]
    crates = []
    for crate in spec.crates:
        origin = point(crate.cell)
        pushes = []
        for push in crate.pushes:
            target = point((push["x"], push["y"]))
            pushes.append(push_record(origin, target, push.get("onto_plate")))
        crates.append(Crate(crate.id, origin, pushes, [point(cell) for cell in crate.stand_cells]))
    goals = [Goal(goal.id, point(goal.cell), goal.agent, side_map[goal.side]) for goal in spec.goals]
    return Spec(
        map_id=spec.map_id,
        seed=spec.seed,
        map_type=spec.map_type,
        width=width,
        height=height,
        axis="y" if spec.axis == "x" else "x",
        start_side=side_map[spec.start_side],
        goal_side=side_map[spec.goal_side],
        grid=grid,
        plates=plates,
        doors=doors,
        crates=crates,
        goals=goals,
        agent1=point(spec.agent1),
        agent2=point(spec.agent2),
        separated=spec.separated,
        independent=spec.independent,
        clear_condition=spec.clear_condition,
        regions=spec.regions,
    )


def finish_crate(
    grid: list[list[str]],
    width: int,
    height: int,
    axis: str,
    start_side: str,
    goal_side: str,
    agents: tuple[tuple[int, int], tuple[int, int]],
    goal: tuple[int, int],
    plates: list[Plate],
    doors: list[Door],
    crates: list[Crate],
    separated: bool,
    map_id: str,
    seed: int,
    map_type: str,
) -> Spec:
    for plate in plates:
        grid[plate.cell[1]][plate.cell[0]] = PLATE
    for crate in crates:
        grid[crate.cell[1]][crate.cell[0]] = OBJECT
    grid[goal[1]][goal[0]] = GOAL
    grid[agents[0][1]][agents[0][0]] = AGENT1
    grid[agents[1][1]][agents[1][0]] = AGENT2
    return Spec(
        map_id=map_id,
        seed=seed,
        map_type=map_type,
        width=width,
        height=height,
        axis=axis,
        start_side=start_side,
        goal_side=goal_side,
        grid=grid,
        plates=plates,
        doors=doors,
        crates=crates,
        goals=[Goal("goal_0", goal, "both", goal_side)],
        agent1=agents[0],
        agent2=agents[1],
        separated=separated,
        independent=False,
        clear_condition="all_agents_reach_goal",
    )


def build_crate_straight(seed: int, map_id: str) -> Spec | None:
    rng = random.Random(seed)
    width = rng.choice((20, 22, 24))
    height = rng.choice((16, 18))
    y0 = rng.randint(1, height - 10)
    lane_x = rng.randint(4, 7)
    door_x = rng.randint(max(lane_x + 3, 11), width - 5)
    grid = blank(width, height)
    carve_rect(grid, 1, y0, width - 2, y0 + 2)
    if rng.random() < 0.65:
        carve_rect(grid, door_x + 1, y0, width - 2, min(height - 2, y0 + 4))
    painted = paint_door(grid, (door_x, y0 + 1), (0, 1))
    if painted is None:
        return None
    door_cell, door_span = painted
    behind = (lane_x, y0 + 1)
    assist = (lane_x + 1, y0 + 1)
    placed = carve_coop_lane(grid, behind, assist, (0, 1))
    if placed is None:
        return None
    box, plate = placed
    for y in range(y0 + 3, height - 1):
        if grid[y][door_x] == FLOOR:
            grid[y][door_x] = WALL
    agent1 = (1, y0)
    agent2 = (2, y0)
    spec = finish_crate(
        grid, width, height, "x", "west", "east", (agent1, agent2), (width - 2, y0 + 1),
        [Plate("plate_0", plate, ["door_0"], 0)],
        [Door("door_0", door_cell, ["plate_0"], door_span)],
        [Crate("object_0", box, [push_record(box, plate, "plate_0")], [behind, assist])],
        True, map_id, seed, "crate_straight",
    )
    if rng.random() < 0.5:
        spec = transpose(spec)
        spec.map_type = "crate_straight"
    return spec


def build_crate_bend(seed: int, map_id: str) -> Spec | None:
    rng = random.Random(seed)
    width = rng.choice((18, 20, 22))
    height = rng.choice((17, 19))
    top = 3
    shaft_w = 3
    east = rng.random() < 0.5
    grid = blank(width, height)
    carve_rect(grid, 1, 1, width - 2, top)
    if east:
        shaft_x0 = width - 1 - shaft_w
        carve_rect(grid, shaft_x0, 1, width - 2, height - 2)
        direction = -1
    else:
        shaft_x0 = 1
        carve_rect(grid, 1, 1, shaft_w, height - 2)
        direction = 1
    door_y = rng.randint(top + 5, height - 5)
    shaft_xs = list(range(shaft_x0, shaft_x0 + shaft_w))
    painted = paint_door(grid, (shaft_xs[1], door_y), (1, 0))
    if painted is None:
        return None
    door_cell, door_span = painted
    edge = shaft_xs[0] if east else shaft_xs[-1]
    behind = (edge + direction, door_y - 2)
    assist = (edge + direction, door_y - 3)
    placed = carve_coop_lane(grid, behind, assist, (direction, 0))
    if placed is None:
        return None
    box, plate = placed
    goal = (shaft_xs[1], height - 2)
    agent1 = (3 if east else width - 4, 2)
    agent2 = (6 if east else width - 7, top)
    if not inside(grid, *agent2) or grid[agent2[1]][agent2[0]] != FLOOR:
        return None
    if agent1 in (box, plate, behind, assist) or agent2 in (box, plate, behind, assist):
        return None
    spec = finish_crate(
        grid, width, height, "y", "north", "south", (agent1, agent2), goal,
        [Plate("plate_0", plate, ["door_0"], 0)],
        [Door("door_0", door_cell, ["plate_0"], door_span)],
        [Crate("object_0", box, [push_record(box, plate, "plate_0")], [behind, assist])],
        abs(agent1[0] - agent2[0]) >= 3, map_id, seed, "crate_bend",
    )
    return spec


def build_crate_s(seed: int, map_id: str) -> Spec | None:
    rng = random.Random(seed)
    gap = 5
    lane = 3
    y0 = 1
    y1 = y0 + lane + gap
    y2 = y1 + lane + gap
    height = y2 + lane + 2
    width = rng.choice((24, 26, 28))
    door_x = rng.randint(12, 15)
    grid = blank(width, height)
    carve_rect(grid, 1, y0, width - 2, y0 + lane - 1)
    carve_rect(grid, 1, y1, width - 2, y1 + lane - 1)
    carve_rect(grid, 1, y2, width - 2, y2 + lane - 1)
    carve_rect(grid, width - 4, y0, width - 2, y1 + lane - 1)
    carve_rect(grid, 1, y1, 3, y2 + lane - 1)
    painted = paint_door(grid, (door_x, y0 + 1), (0, 1))
    if painted is None:
        return None
    door_cell, door_span = painted
    behind = (door_x - 6, y0 + 1)
    assist = (door_x - 5, y0 + 1)
    placed = carve_coop_lane(grid, behind, assist, (0, 1))
    if placed is None or placed[1][1] >= y1:
        return None
    box, plate = placed
    agent1 = (2, y0)
    agent2 = (3, y0 + 1)
    goal = (width - 3, y2 + 1)
    spec = finish_crate(
        grid, width, height, "x", "west", "east", (agent1, agent2), goal,
        [Plate("plate_0", plate, ["door_0"], 0)],
        [Door("door_0", door_cell, ["plate_0"], door_span)],
        [Crate("object_0", box, [push_record(box, plate, "plate_0")], [behind, assist])],
        True, map_id, seed, "crate_s",
    )
    if rng.random() < 0.5:
        spec = transpose(spec)
    return spec


def build_crate_chain(seed: int, map_id: str) -> Spec | None:
    rng = random.Random(seed)
    width = rng.choice((32, 34))
    height = 16
    y0 = 2
    grid = blank(width, height)
    carve_rect(grid, 1, y0, width - 2, y0 + 2)
    doors = []
    plates = []
    crates = []
    for index, (lane_x, door_x) in enumerate(((4, 13), (18, 26))):
        painted = paint_door(grid, (door_x, y0 + 1), (0, 1))
        if painted is None:
            return None
        door_cell, door_span = painted
        behind = (lane_x, y0 + 1)
        assist = (lane_x + 1, y0 + 1)
        placed = carve_coop_lane(grid, behind, assist, (0, 1))
        if placed is None:
            return None
        box, plate = placed
        for y in range(y0 + 3, height - 1):
            if grid[y][door_x] == FLOOR:
                grid[y][door_x] = WALL
        plate_id = f"plate_{index}"
        door_id = f"door_{index}"
        plates.append(Plate(plate_id, plate, [door_id], 0))
        doors.append(Door(door_id, door_cell, [plate_id], door_span))
        crates.append(Crate(f"object_{index}", box, [push_record(box, plate, plate_id)], [behind, assist]))
    spec = finish_crate(
        grid, width, height, "x", "west", "east", ((2, y0 + 1), (4, y0 + 1)), (width - 3, y0 + 1),
        plates, doors, crates, True, map_id, seed, "crate_chain",
    )
    if rng.random() < 0.5:
        spec = transpose(spec)
    return spec


def build_split(seed: int, map_id: str) -> Spec | None:
    """Two sealed rooms. Each plate latches only the other agent's door."""
    rng = random.Random(seed)
    width = rng.choice((24, 26))
    height = 18
    grid = blank(width, height)
    left_x1 = width // 2 - 2
    right_x0 = width // 2 + 1
    door_y = 10
    regions = (
        (1, left_x1, rng.randint(3, 5)),
        (right_x0, width - 2, rng.randint(width // 2 + 3, width - 5)),
    )
    plates: list[Plate] = []
    doors: list[Door] = []
    goals: list[tuple[int, int]] = []
    for index, (x0, x1, door_x) in enumerate(regions):
        if not (x0 + 1 < door_x < x1 - 1):
            return None
        carve_rect(grid, x0, 1, x1, door_y - 1)
        carve_rect(grid, x0, door_y + 1, x1, height - 2)
        for x in range(x0, x1 + 1):
            grid[door_y][x] = WALL
        painted = paint_door(grid, (door_x, door_y), (1, 0))
        if painted is None:
            return None
        door_cell, door_span = painted
        # One plate sits on the approach side of this door, with a gap before the door.
        plate = (door_x, door_y - 3)
        plates.append(Plate(f"plate_{index}", plate, [], index))
        doors.append(Door(f"door_{index}", door_cell, [], door_span))
        goals.append((door_x, height - 2))
    agent1 = (regions[0][2], 1)
    agent2 = (regions[1][2], 1)
    if len({agent1, agent2, plates[0].cell, plates[1].cell, goals[0], goals[1]}) < 6:
        return None
    both_doors = [door.id for door in doors]
    both_plates = [plate.id for plate in plates]
    for plate in plates:
        plate.opens = list(both_doors)
    for door in doors:
        door.opened_by = list(both_plates)
    for plate in plates:
        grid[plate.cell[1]][plate.cell[0]] = PLATE
    for goal in goals:
        grid[goal[1]][goal[0]] = GOAL
    grid[agent1[1]][agent1[0]] = AGENT1
    grid[agent2[1]][agent2[0]] = AGENT2
    spec = Spec(
        map_id=map_id,
        seed=seed,
        map_type="split_vertical",
        width=width,
        height=height,
        axis="y",
        start_side="north",
        goal_side="south",
        grid=grid,
        plates=plates,
        doors=doors,
        crates=[],
        goals=[
            Goal("goal_1", goals[0], "1", "south"),
            Goal("goal_2", goals[1], "2", "south"),
        ],
        agent1=agent1,
        agent2=agent2,
        separated=True,
        independent=True,
        clear_condition="each_agent_reaches_own_goal",
        regions={"agent1": 0, "agent2": 1},
    )
    if rng.random() < 0.5:
        spec = transpose(spec)
        spec.map_type = "split_horizontal"
    return spec


def build_paired(seed: int, map_id: str) -> Spec | None:
    rng = random.Random(seed)
    if rng.random() < 0.5:
        return build_paired_offset(rng, seed, map_id)
    return build_paired_l(rng, seed, map_id)


def build_paired_offset(rng: random.Random, seed: int, map_id: str) -> Spec | None:
    width, height = rng.choice(((21, 15), (23, 16)))
    sx1 = rng.randint(7, 9)
    sy1 = rng.randint(9, 11)
    gy0 = rng.randint(4, 6)
    if gy0 > sy1 - 3:
        gy0 = sy1 - 3
    grid = blank(width, height)
    carve_rect(grid, 1, 1, sx1, sy1)
    carve_rect(grid, sx1 + 2, gy0, width - 2, height - 2)
    if rng.random() < 0.7:
        carve_rect(grid, sx1 - 2, 1, sx1, 3)
        for y in range(1, 4):
            for x in range(sx1 - 2, sx1 + 1):
                grid[y][x] = WALL
    door_x = sx1 + 1
    overlap = [y for y in range(max(gy0, 1), min(sy1, height - 2) + 1)]
    if len(overlap) < 5:
        return None
    door_y = overlap[len(overlap) // 2]
    painted = paint_door(grid, (door_x, door_y), (0, 1))
    if painted is None:
        return None
    door_cell, door_span = painted
    plate_a = (2, sy1 - 1)
    plate_b = (min(width - 3, sx1 + 6), min(height - 3, gy0 + 4))
    goal = (width - 2, height - 2)
    agent1 = (2, 2)
    agent2 = (3, 2)
    box = (5, sy1 - 1)
    behind = (5, sy1 - 2)
    assist = (6, sy1 - 2)
    for cell in (plate_a, plate_b, goal, agent1, agent2, box, behind, assist, (door_x - 1, door_y), (door_x + 1, door_y)):
        if not inside(grid, *cell) or grid[cell[1]][cell[0]] == WALL:
            return None
    if manhattan(box, plate_a) < MIN_PLATE_OBJECT_DISTANCE or manhattan(box, plate_b) < MIN_PLATE_OBJECT_DISTANCE:
        return None
    spec = finish_crate(
        grid, width, height, "x", "west", "east", (agent1, agent2), goal,
        [Plate("plate_0", plate_a, ["door_0"], 0), Plate("plate_1", plate_b, ["door_0"], 1)],
        [Door("door_0", door_cell, ["plate_0", "plate_1"], door_span)],
        [Crate("object_0", box, [push_record(box, (box[0], box[1] - 1), None)], [behind, assist])],
        True, map_id, seed, "paired_offset",
    )
    spec.clear_condition = "all_agents_reach_goal"
    if rng.random() < 0.5:
        spec = transpose(spec)
        spec.map_type = "paired_offset"
    return spec


def build_paired_l(rng: random.Random, seed: int, map_id: str) -> Spec | None:
    width, height = 24, 18
    grid = blank(width, height)
    carve_rect(grid, 1, 1, 3, 15)
    carve_rect(grid, 1, 12, 9, 15)
    carve_rect(grid, 11, 8, width - 2, 16)
    painted = paint_door(grid, (10, 13), (0, 1))
    if painted is None:
        return None
    door_cell, door_span = painted
    plate_a = (2, 9)
    plate_b = (16, 12)
    box = (2, 4)
    behind = (2, 3)
    assist = (1, 3)
    goal = (width - 2, 13)
    agent1 = (2, 2)
    agent2 = (2, 6)
    spec = finish_crate(
        grid, width, height, "x", "west", "east", (agent1, agent2), goal,
        [Plate("plate_0", plate_a, ["door_0"], 0), Plate("plate_1", plate_b, ["door_0"], 1)],
        [Door("door_0", door_cell, ["plate_0", "plate_1"], door_span)],
        [Crate("object_0", box, [push_record(box, (2, 5), None)], [behind, assist])],
        True, map_id, seed, "paired_l",
    )
    if rng.random() < 0.5:
        spec = transpose(spec)
        spec.map_type = "paired_l"
    return spec


BUILDERS = [
    build_crate_straight,
    build_crate_bend,
    build_crate_s,
    build_crate_chain,
    build_split,
    build_paired,
]


def common_meta(spec: Spec) -> dict:
    return {
        "map_id": spec.map_id,
        "seed": spec.seed,
        "map_type": spec.map_type,
        "note": TYPE_NOTES[spec.map_type],
        "width": spec.width,
        "height": spec.height,
        "cell_size": CELL,
        "origin": "top_left",
        "axes": {"x": "right", "y": "down"},
        "start_side": spec.start_side,
        "goal_side": spec.goal_side,
        "independent_regions": spec.independent,
        "clear_condition": spec.clear_condition,
    }


def feature_files(spec: Spec) -> dict[str, dict]:
    meta = common_meta(spec)
    walls = [[x, y] for y, row in enumerate(spec.grid) for x, value in enumerate(row) if value == WALL]
    cross = spec.independent
    blocks, pressed_by = split_roles(spec) if cross else ({}, {})
    other = {"1": "2", "2": "1"}
    plate_mode = "latch" if cross else "hold"
    door_mode = "latch" if cross else "hold"
    return {
        "walls.yaml": {
            **meta,
            "feature": WALL,
            "name": TILES[WALL]["name"],
            "asset": "assets/tiles/wall.png",
            "color": list(TILES[WALL]["color"]),
            "cells": walls,
        },
        "plates.yaml": {
            **meta,
            "feature": PLATE,
            "name": TILES[PLATE]["name"],
            "asset": "assets/tiles/plate.png",
            "color": list(TILES[PLATE]["color"]),
            "mode": plate_mode,
            "occupied_by": ["agent"] if cross else ["agent", "movable_object"],
            "plates": [
                {
                    "id": plate.id,
                    "x": plate.cell[0],
                    "y": plate.cell[1],
                    "region": plate.region,
                    "opens": plate.opens,
                    "mode": plate_mode,
                    **(
                        {
                            "pressed_by_agent": pressed_by[plate.id],
                            "required_for_doors": plate.opens,
                        }
                        if cross
                        else {}
                    ),
                }
                for plate in spec.plates
            ],
        },
        "doors.yaml": {
            **meta,
            "feature": DOOR,
            "name": TILES[DOOR]["name"],
            "asset": "assets/tiles/door.png",
            "color": list(TILES[DOOR]["color"]),
            "logic": "all" if cross else "any",
            "mode": door_mode,
            "description": (
                "각 문 앞 발판을 포함해 연결된 발판이 모두 한 번씩 밟히면 열린 채로 유지된다. 한쪽만 밟아서는 나갈 수 없다."
                if cross
                else "연결된 발판 중 하나라도 에이전트나 물건이 올려져 있는 동안 열린다"
            ),
            "doors": [
                {
                    "id": door.id,
                    "x": door.cell[0],
                    "y": door.cell[1],
                    "cells": [[cell[0], cell[1]] for cell in (door.cell, *door.span)],
                    "opened_by": door.opened_by,
                    "logic": "all" if cross else "any",
                    "mode": door_mode,
                    "initial_open": False,
                    **(
                        {
                            "blocks_agent": blocks[door.id],
                            "opened_by_agents": ["1", "2"],
                        }
                        if cross
                        else {}
                    ),
                }
                for door in spec.doors
            ],
        },
        "objects.yaml": {
            **meta,
            "feature": OBJECT,
            "name": TILES[OBJECT]["name"],
            "asset": "assets/tiles/object.png",
            "color": list(TILES[OBJECT]["color"]),
            "objects": [
                {
                    "id": crate.id,
                    "x": crate.cell[0],
                    "y": crate.cell[1],
                    "pushable": True,
                    "requires_both_agents": True,
                    "pushes_onto_plate": next((push.get("onto_plate") for push in crate.pushes if push.get("onto_plate")), None),
                    "plate_distance": min(manhattan(crate.cell, plate.cell) for plate in spec.plates),
                    "stand_cells": [[cell[0], cell[1]] for cell in crate.stand_cells],
                    "push_neighbors": crate.pushes,
                }
                for crate in spec.crates
            ],
        },
        "goal.yaml": {
            **meta,
            "feature": GOAL,
            "name": TILES[GOAL]["name"],
            "asset": "assets/tiles/goal.png",
            "color": list(TILES[GOAL]["color"]),
            "x": spec.goals[0].cell[0],
            "y": spec.goals[0].cell[1],
            "side": spec.goals[0].side,
            "goals": [
                {
                    "id": goal.id,
                    "x": goal.cell[0],
                    "y": goal.cell[1],
                    "agent": goal.agent,
                    "side": goal.side,
                    **(
                        {"reachable_alone": False, "requires_agent": other[goal.agent]}
                        if cross
                        else {}
                    ),
                }
                for goal in spec.goals
            ],
        },
        "agent1_start.yaml": {
            **meta,
            "feature": AGENT1,
            "name": TILES[AGENT1]["name"],
            "asset": "assets/tiles/agent1_start.png",
            "color": list(TILES[AGENT1]["color"]),
            "x": spec.agent1[0],
            "y": spec.agent1[1],
            "side": spec.start_side,
            "region": spec.regions.get("agent1", 0),
            "separated_from_agent2": spec.separated,
        },
        "agent2_start.yaml": {
            **meta,
            "feature": AGENT2,
            "name": TILES[AGENT2]["name"],
            "asset": "assets/tiles/agent2_start.png",
            "color": list(TILES[AGENT2]["color"]),
            "x": spec.agent2[0],
            "y": spec.agent2[1],
            "side": spec.start_side,
            "region": spec.regions.get("agent2", 0),
            "separated_from_agent1": spec.separated,
        },
    }


def render(spec: Spec, tiles: dict[str, Image.Image], path: Path) -> None:
    canvas = Image.new("RGB", (spec.width * CELL, spec.height * CELL))
    for y, row in enumerate(spec.grid):
        for x, value in enumerate(row):
            canvas.paste(tiles[value], (x * CELL, y * CELL))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    sx, sy = SAMPLE
    for y, row in enumerate(spec.grid):
        for x, value in enumerate(row):
            pixel = canvas.getpixel((x * CELL + sx, y * CELL + sy))
            if pixel != TILES[value]["color"]:
                raise AssertionError(f"{value} tile was not pasted at {(x, y)}")


def try_build(builder, seed: int, map_id: str) -> Spec | None:
    try:
        spec = builder(seed, map_id)
    except (ValueError, IndexError, AssertionError):
        return None
    if spec is None:
        return None
    try:
        validate(spec)
    except AssertionError:
        return None
    return spec


def generate(count: int, seed: int, out_dir: Path) -> None:
    draw_tiles()
    tiles = load_tiles()
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    cursor = seed
    for made in range(count):
        builder = BUILDERS[made % len(BUILDERS)]
        spec = None
        while spec is None:
            spec = try_build(builder, cursor, f"{made:05d}")
            cursor += 1
            if cursor - seed > count * 40:
                raise RuntimeError(f"rejected too many maps at {builder.__name__}")
        folder = out_dir / spec.map_id
        render(spec, tiles, folder / "map.png")
        for name, payload in feature_files(spec).items():
            write_yaml(folder / name, payload)
        index.append(
            {
                "id": spec.map_id,
                "seed": spec.seed,
                "map_type": spec.map_type,
                "note": TYPE_NOTES[spec.map_type],
                "png": (folder / "map.png").relative_to(ROOT).as_posix(),
                "width": spec.width,
                "height": spec.height,
                "start_side": spec.start_side,
                "goal_side": spec.goal_side,
                "independent_regions": spec.independent,
                "clear_condition": spec.clear_condition,
                "n_plates": len(spec.plates),
                "n_doors": len(spec.doors),
                "n_objects": len(spec.crates),
            }
        )
    write_yaml(
        out_dir / "index.yaml",
        {"cell_size": CELL, "count": count, "floor_color": [255, 255, 255], "maps": index},
    )
    print(f"maps: {out_dir} ({count})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cooperative escape training maps.")
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=ROOT / "maps" / "train")
    args = parser.parse_args()
    generate(args.count, args.seed, args.out)


if __name__ == "__main__":
    main()
