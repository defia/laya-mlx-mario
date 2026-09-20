"""Decision layer: the original remote TypeSafe/Jev policy replaced by a
local laya-mlx Agent. One `predict` call answers the same three typed
judgments (choice / noul / score) in ~10ms with zero network traffic.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .actions import Action
from .describe import build_questions, describe
from .state import MarioSnapshot


@dataclass(frozen=True)
class Decision:
    action: Action
    confidence: float
    probabilities: Mapping[str, float]
    latency_ms: float
    jump_needed_probability: float | None = None
    danger_score: float | None = None
    prompt: str = ""  # the exact state text the local model read
    labels: Mapping[str, str] | None = None  # option labels the model chose among
    shielded: bool = False  # geometric veto: the model's pick was provably fatal
    rescue: bool = False  # action came from the stuck-at-wall maneuver


# measured on the emulator: a full-hold ground jump stays airborne ~44 frames
JUMP_AIRTIME_FRAMES = 44
RUN_FLIGHT_SPEED = 2.5  # px/frame a B-button jump sustains even from a walk
DECISION_HORIZON = 8  # frames_per_decision; forward macros advance this much


def fatal_actions(snapshot: MarioSnapshot) -> set[Action]:
    """Actions provably fatal this decision, from geometry alone.

    - walking into the gap within this macro;
    - a jump whose landing point (airtime x speed) falls inside the gap;
    - walking into an enemy that makes contact within this macro.
    """
    bad: set[Action] = set()
    if snapshot.grounded:
        hazard = snapshot.threat_features()
        contact = hazard.get("estimated_contact_frames")
        if hazard.get("enemy_ahead") and contact is not None and 0 < contact:
            # a takeoff needs ~8 frames of rise to clear enemy height, and a
            # decision macro is 8 more: allowing a walk at contact ~16 burns
            # the last window (measured death: forced takeoff at contact 6
            # collided at low altitude). Veto walking from contact 16 down.
            takeoff_deadline = contact <= DECISION_HORIZON + 8
            if takeoff_deadline:
                bad |= {Action.RIGHT, Action.RIGHT_RUN, Action.NOOP}
            elif contact <= 30:
                pass  # prime takeoff window: jumping now is genuinely safe
            elif contact <= 48:
                # hopping from here crosses low on the descent — a measured
                # side-hit death; walk in, the window opens next decisions
                bad |= {Action.RIGHT_JUMP, Action.RIGHT_RUN_JUMP}
        # symmetric blind spot: an enemy closing from BEHIND kills a Mario
        # who stands still (measured: stuck at the wall, goomba pair walked
        # into his back while the rescue waited). A hop lets it pass below.
        for enemy in snapshot.enemies:
            if (
                -16 <= enemy.dx_pixels < 0
                and enemy.relative_velocity_x > 0  # approaching from behind
            ):
                bad |= {Action.RIGHT, Action.RIGHT_RUN, Action.NOOP}
        terrain = snapshot.navigation_features()
        if terrain.get("geometry_available"):
            bad |= _fatal_gap_actions(snapshot, terrain)
        return bad
    # airborne: switching buttons mid-jump truncates the arc — the measured
    # death was right_jump -> jump at 60px out, dropping Mario onto the enemy
    hazard = snapshot.threat_features()
    contact = hazard.get("estimated_contact_frames")
    if (
        hazard.get("enemy_ahead")
        and contact is not None
        and 0 < contact <= 48
        and not snapshot.crossing_gap
        and snapshot.airborne_frames <= 40
    ):
        bad |= {Action.NOOP, Action.RIGHT, Action.RIGHT_RUN, Action.JUMP, Action.LEFT}
        # right_jump / right_run_jump stay admissible: complete the arc
    return bad


def _fatal_gap_actions(
    snapshot: MarioSnapshot, terrain: dict
) -> set[Action]:
    distance = terrain.get("gap_distance_tiles")
    if distance is None or distance > 6:
        return set()
    # an obstacle at or before the gap = ground hidden behind a pipe/wall,
    # not a walkable gap: vetoing jumps there breaks locomotion (measured:
    # phantom 8-tile gap at the first pipe froze Mario two pipes later)
    obstacle = terrain.get("obstacle_distance_tiles")
    if obstacle is not None and obstacle <= distance:
        return set()
    width = min(terrain.get("gap_width_tiles_visible", 0) or 1, 4)  # streamed-page cap
    speed = max(snapshot.dx, 1)

    bad: set[Action] = set()
    if distance * 16 <= DECISION_HORIZON * speed + 8:  # reaches the edge this macro
        bad |= {Action.RIGHT, Action.RIGHT_RUN}
    for action, flight_speed in (
        (Action.RIGHT_JUMP, speed),
        (Action.JUMP, 1),
        (Action.RIGHT_RUN_JUMP, max(speed, RUN_FLIGHT_SPEED)),
    ):
        reach = JUMP_AIRTIME_FRAMES * flight_speed / 16
        if distance < reach < distance + width + 0.5:  # lands inside the gap
            bad.add(action)
    return bad


def _fallback(bad: set[Action]) -> Action:
    """Preferred admissible action when the model's pick is vetoed."""
    for candidate in (
        Action.RIGHT_RUN_JUMP,
        Action.RIGHT_JUMP,
        Action.RIGHT_RUN,
        Action.RIGHT,
        Action.LEFT,
        Action.JUMP,
        Action.NOOP,
    ):
        if candidate not in bad:
            return candidate
    return Action.NOOP


# Stuck-at-tall-wall rescue (the 898 wall): a standing hop cannot clear 4+
# tiles and the model never retreats on its own (P(left) measured at 0.02
# under every wording). Fixed maneuver: back off a few steps, build run
# speed, then a running jump holds through the arc.
RESCUE_PLAN: tuple[Action, ...] = (
    Action.LEFT,
    Action.LEFT,
    Action.LEFT,
    Action.LEFT,
    Action.RIGHT_RUN,
    Action.RIGHT_RUN,
    Action.RIGHT_RUN,
    Action.RIGHT_RUN_JUMP,
    Action.RIGHT_JUMP,
    Action.RIGHT_JUMP,
)


def stuck_at_wall(snapshot: MarioSnapshot) -> bool:
    if not snapshot.grounded or snapshot.stalled_steps < 4:
        return False
    terrain = snapshot.navigation_features()
    if not terrain.get("geometry_available"):
        return False
    distance = terrain.get("obstacle_distance_tiles")
    height = terrain.get("obstacle_height_tiles") or 0
    if distance is None or distance > 2 or height < 4:
        return False
    # only an enemy BEHIND blocks the retreat (we would back into it); ahead
    # enemies get FARTHER as we retreat, so they are fine — the original
    # 112px-any-side guard let a passing goomba pair freeze the rescue until
    # they walked into a Mario stuck at the wall
    return not any(-80 <= e.dx_pixels < 0 for e in snapshot.enemies)


class Policy:
    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision: ...


class LayaPolicy:
    """Local laya-mlx policy; typed questions, no network.

    mode="pure" (the research target): no shield, no rescue maneuver, no
    verdict tags on the options — only state facts plus constant game
    background; the model's own judgment is the whole policy.
    mode="assist": the documented 1518 baseline — geometric shield, rescue
    maneuver and verdict tags all on.
    """

    def __init__(
        self,
        model: str = "models/hub/laya-multilingual-mlx",
        dtype: str = "float16",
        lang: str = "zh",
        mode: str = "pure",
        background: bool | None = None,
    ) -> None:
        from laya_mlx import Agent

        if mode not in ("pure", "assist"):
            raise ValueError(f"mode must be 'pure' or 'assist', got {mode!r}")
        self._agent = Agent(model, dtype=dtype)
        self._lang = lang
        self._shield = mode == "assist"
        self._verdicts = mode == "assist"
        self._rescue = mode == "assist"
        self._background = (mode == "pure") if background is None else background
        self._plan: list[Action] = []
        self.rescues = 0  # completed rescue-maneuver starts (readable by the runner)

    def close(self) -> None:
        close = getattr(self._agent, "close", None)
        if callable(close):
            close()

    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        action_tuple = tuple(actions)
        prompt = describe(snapshot, self._lang, background=self._background)
        questions = build_questions(snapshot, action_tuple, self._lang, verdicts=self._verdicts)
        started = time.perf_counter()
        result = self._agent.predict(prompt, questions)
        latency_ms = (time.perf_counter() - started) * 1000.0
        answers = result["answers"]

        action_answer = answers["next_action"]
        probabilities = {
            str(key): float(value)
            for key, value in dict(action_answer["probabilities"]).items()
        }
        choice = str(action_answer["choice"])
        valid = {action.value: action for action in action_tuple}
        picked = valid.get(choice) or valid[max(probabilities, key=probabilities.get)]

        bad = fatal_actions(snapshot)
        action, shielded, rescue = picked, False, False
        if self._shield and picked in bad:
            action = _fallback(bad)
            shielded = True
        # stuck-at-wall maneuver takes precedence over everything; the model
        # still answers (probabilities shown honestly), only the macro differs
        if self._rescue and not self._plan and stuck_at_wall(snapshot):
            self._plan = list(RESCUE_PLAN)
            self.rescues += 1
        if self._plan:
            action = self._plan.pop(0)
            shielded, rescue = True, True
        return Decision(
            action=action,
            confidence=float(action_answer.get("confidence", 0.0)),
            probabilities=probabilities,
            latency_ms=latency_ms,
            jump_needed_probability=float(answers["jump_needed"]["noul"]),
            danger_score=float(answers["danger"]["score"]),
            prompt=prompt,
            labels=dict(questions["next_action"]["criteria"]),
            shielded=shielded,
            rescue=rescue,
        )


class HeuristicPolicy:
    """Offline smoke-test policy; not intended as the Mario benchmark baseline."""

    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        allowed = set(actions)
        action = Action.RIGHT_RUN if Action.RIGHT_RUN in allowed else actions[0]
        if snapshot.stalled_steps >= 2 and Action.RIGHT_RUN_JUMP in allowed:
            action = Action.RIGHT_RUN_JUMP
        return Decision(
            action=action,
            confidence=1.0,
            probabilities={candidate.value: float(candidate == action) for candidate in actions},
            latency_ms=0.0,
        )
