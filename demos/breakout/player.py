"""One Laya decision per step: action choice, drop-risk estimate, urgency score.

Feature-assisted semantics as upstream: the geometric planner predicts the
ball's landing column and the set of admissible actions (after that move, the
follow-the-ball fallback still catches the ball). The model genuinely chooses
among them; a pick outside the set is corrected and marked as a shield
intervention. The model's own probabilities are always shown.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .describe import build_questions, describe
from .game import (
    BRICK_ROWS,
    BRICK_TOP,
    PADDLE_ROW,
    PADDLE_SPEED,
    PADDLE_W,
    BreakoutGame,
)

ACTIONS = ("left", "stay", "right")
_DIR = {"left": -1, "stay": 0, "right": 1}
CRITICAL_ETA = 8  # within this many steps, wandering away is no longer admissible


@dataclass
class Decision:
    action: str  # "left" | "stay" | "right" (post-shield)
    raw_action: str  # model's own pick
    probabilities: dict[str, float]
    danger: float  # P(ball drops if the paddle stays still)
    urgency: float  # expected rubric level 0..3
    inference_ms: float
    shielded: bool = False
    prompt: str = ""  # state text the model actually read
    landing: int | None = None  # predicted landing column
    eta: int = 0  # steps until the ball arrives
    gains: dict[str, int] | None = None  # steer: rally points per receive point


def _rally_gain(game: BreakoutGame, target_col: int, cap: int = 150) -> int:
    """Combo-weighted points of the next rally if the paddle simply parks with
    its center at target_col: let the ball arrive, bounce, and score until it
    returns to the paddle (or drops). Uses the real tick, so the physics can
    never drift from reality; `score` is monotonic, so level refills are safe."""
    scratch = game.clone()
    scratch.last_event = ""
    scratch.px = max(0, min(game.cols - PADDLE_W, target_col - PADDLE_W // 2))
    start_score = None
    for _ in range(cap):
        scratch.tick()
        if scratch.last_event == "paddle":
            if start_score is None:
                start_score = scratch.score  # the catch: rally starts here
            else:
                break  # ball came back: rally over
        elif start_score is not None and scratch.last_event == "miss":
            break  # parked paddle missed the return: rally over
        if not scratch.alive or scratch.last_event == "clear":
            break
    if start_score is None:
        return 0  # ball did not reach the parked paddle within the horizon
    return scratch.score - start_score


def _target_center(game: BreakoutGame, landing_col: int | None, strategy: str) -> int | None:
    """Paddle center that catches the ball. "steady" aims the paddle center at
    the receive point; "steer" tries every reachable receive point (center /
    either edge) by parking there in simulation and keeps the one whose next
    rally scores the most, preferring the plain center hit on ties."""
    if landing_col is None:
        return None
    if strategy != "steer":
        return landing_col
    lo, hi = PADDLE_W // 2, game.cols - 1 - PADDLE_W // 2
    # a center hit mirrors the ball; edge hits force their side.
    # edge targets use the paddle CORNER (±3): even with a column of placement
    # slop the ball still lands in the forced zone, so the model's imprecision
    # cannot silently turn a steer into a mirror.
    # the center option is listed first so ties keep the mirror — early on the
    # corridor it digs is what later rallies cash in as deep combos.
    options = [landing_col]
    if landing_col + 3 <= hi:
        options.append(landing_col + 3)  # ball leaves to the left
    if landing_col - 3 >= lo:
        options.append(landing_col - 3)  # ball leaves to the right
    best_col, best_gain = landing_col, -1
    for col in options:
        gain = _rally_gain(game, col)
        if gain > best_gain:
            best_col, best_gain = col, gain
    return best_col


def _policy_from(game: BreakoutGame, landing_col: int | None, strategy: str) -> str:
    if landing_col is None:
        return "stay"
    dx = _target_center(game, landing_col, strategy) - game.paddle_center
    if dx < -1:
        return "left"
    if dx > 1:
        return "right"
    return "stay"


def geometric_policy(game: BreakoutGame, strategy: str = "steady") -> str:
    """Greedy fallback: move toward the (strategy-adjusted) landing point."""
    col, _eta = game.landing()
    return _policy_from(game, col, strategy)


def _descent_aim(scratch: BreakoutGame) -> int:
    """Column the ball will cross the paddle row at, when it is already
    descending below the brick wall — only side walls can still deflect it,
    so this is a cheap closed-form fold. Above the wall we just shadow the
    ball (its final approach is not determined yet)."""
    if scratch.vy > 0 and scratch.ball_y > BRICK_TOP + BRICK_ROWS - 1:
        x, vx = scratch.ball_x, scratch.vx
        for _ in range(PADDLE_ROW - scratch.ball_y):
            x += vx
            if x < 0 or x > scratch.cols - 1:
                vx = -vx
        return x
    return scratch.ball_x


def _catches(game: BreakoutGame, action: str, horizon: int = 260) -> bool:
    """Play `action` through its own tick, then let the follow-the-ball
    fallback chase; admissible iff the ball is still caught (or the wall is
    cleared) before it drops."""
    scratch = game.clone()
    scratch.last_event = ""
    scratch.move(action)
    scratch.tick()
    if scratch.last_event in ("paddle", "clear"):
        return True
    if scratch.last_event == "miss" or not scratch.alive:
        return False
    for _ in range(horizon - 1):
        dx = _descent_aim(scratch) - scratch.paddle_center
        scratch.move("left" if dx < 0 else ("right" if dx > 0 else "stay"))
        scratch.tick()
        if scratch.last_event in ("paddle", "clear"):
            return True
        if scratch.last_event == "miss" or not scratch.alive:
            return False
    return True  # ball still cycling far above: nothing imminent


def admissible(game: BreakoutGame, strategy: str = "steady") -> set[str]:
    """Moves after which the ball is still catchable (upstream Snake semantics).

    Endgame band (like Flappy's glide ceiling): once the ball is close, only
    moves that close the distance to the receive point — or keep the paddle
    aligned — stay admissible, so a wrong model pick becomes a shield
    intervention instead of an unrecoverable miss.
    """
    ok = {a for a in ACTIONS if _catches(game, a)}
    col, eta = game.landing()
    if col is not None and eta <= CRITICAL_ETA:
        target = _target_center(game, col, strategy)
        dx0 = target - game.paddle_center
        if abs(dx0) > 3:  # not yet aligned with the receive point

            def dx_after(action: str) -> int:
                return dx0 - _DIR[action] * PADDLE_SPEED

            # only moves that would drift AWAY are banned; holding position
            # stays admissible (the roll-out above still guarantees a catch)
            banded = {a for a in ok if abs(dx_after(a)) <= max(abs(dx0), 3)}
            if banded:
                ok = banded
    return ok


def decide(agent, game: BreakoutGame, shield: bool = True, strategy: str = "steady") -> Decision:
    ok = admissible(game, strategy)
    landing_col, eta = game.landing()
    target_col = _target_center(game, landing_col, strategy) if landing_col is not None else None
    best = _policy_from(game, landing_col, strategy)
    if best not in ok and ok:
        want = _DIR[best]  # keep the "best" label on the admissible action
        best = min(ok, key=lambda a: abs(_DIR[a] - want))  # closest in direction

    prompt = describe(game, ok, best, target_col, eta)
    started = time.perf_counter()
    result = agent.predict(prompt, build_questions(ok, best))
    inference_ms = (time.perf_counter() - started) * 1000.0
    answers = result["answers"]

    raw_action = answers["action"]["choice"]
    probabilities = answers["action"]["probabilities"]
    danger = float(answers["danger"]["noul"])
    urgency = float(answers["urgency"]["score"])

    action, shielded = raw_action, False
    if shield and raw_action not in ok and ok:
        action = best
        shielded = True
        # empty admissible set: nothing catches, execute the model's pick honestly

    gains = None
    if strategy == "steer" and landing_col is not None:
        lo, hi = PADDLE_W // 2, game.cols - 1 - PADDLE_W // 2
        gains = {
            "center": _rally_gain(game, landing_col),
            "left": _rally_gain(game, landing_col + 3) if landing_col + 3 <= hi else -1,
            "right": _rally_gain(game, landing_col - 3) if landing_col - 3 >= lo else -1,
        }

    return Decision(
        action=action,
        raw_action=raw_action,
        probabilities=probabilities,
        danger=danger,
        urgency=urgency,
        inference_ms=inference_ms,
        shielded=shielded,
        prompt=prompt,
        landing=landing_col,
        eta=eta,
        gains=gains,
    )
