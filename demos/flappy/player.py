"""One Laya decision per step: action choice, danger estimate, urgency score.

Following the upstream Snake demo semantics: a geometric planner computes the
set of admissible actions (no immediate death, gap still reachable). The model
genuinely chooses among them; a choice outside the set is corrected and marked
as a shield intervention. The model's original probabilities are always shown.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .describe import STRATEGIES, build_questions, build_questions_pure, describe, describe_pure
from .game import FLAP, FlappyGame


@dataclass
class Decision:
    action: str  # "jump" | "glide" (post-shield)
    raw_action: str  # model's own pick
    probabilities: dict[str, float]
    danger: float  # P(collision if no flap)
    urgency: float  # expected rubric level 0..3
    inference_ms: float
    shielded: bool = False
    prompt: str = ""  # state text the model actually read


def geometric_policy(game: FlappyGame, strategy: str = "lazy") -> str:
    """Greedy fallback toward the strategy's target row."""
    pipe = game.next_pipe
    if pipe is None:
        target = game.rows // 2
    elif strategy == "high":
        target = pipe.gap_top + 1  # hug the upper edge (1 row of margin)
    else:
        target = pipe.gap_center
    ny = game.bird_y + max(-2, min(2, game.vy + 1))
    return "glide" if ny <= target else "jump"


def _survives(game: FlappyGame, action: str, horizon: int = 45) -> bool:
    """Take `action`, then check the geometric fallback still finds a path."""
    scratch = game.clone()
    if action == "jump":
        scratch.flap()
    scratch.tick()
    if not scratch.alive:
        return False
    for _ in range(horizon):
        if geometric_policy(scratch) == "jump":
            scratch.flap()
        scratch.tick()
        if not scratch.alive:
            return False
    return True


def admissible(game: FlappyGame, strategy: str = "lazy") -> set[str]:
    """Actions after which the game is still recoverable (upstream Snake semantics).

    Stricter strategies additionally forbid gliding below their altitude band,
    which is what keeps the bird centered or flying high.
    """
    ok = {action for action in ("jump", "glide") if _survives(game, action)}
    ceiling = STRATEGIES[strategy].get("glide_ceiling")
    pipe = game.next_pipe
    if ceiling and pipe is not None and "glide" in ok:
        ny = game.bird_y + max(-2, min(2, game.vy + 1))
        limit = pipe.gap_center if ceiling == "center" else pipe.gap_top + 1
        if ny > limit:
            ok.discard("glide")
    return ok


def decide(agent, game: FlappyGame, shield: bool = True, strategy: str = "lazy",
           mode: str = "assist") -> Decision:
    """mode="assist": planner verdicts in the labels + optional shield.
    mode="pure": facts only, neutral labels, no shield — the mario pure
    recipe; the model's own safety judgment is the whole policy."""
    if mode == "pure":
        prompt = describe_pure(game)
        started = time.perf_counter()
        result = agent.predict(prompt, build_questions_pure())
        inference_ms = (time.perf_counter() - started) * 1000.0
        answers = result["answers"]
        action_answer = answers["action"]
        return Decision(
            action=str(action_answer["choice"]),
            raw_action=str(action_answer["choice"]),
            probabilities=dict(action_answer["probabilities"]),
            danger=float(answers["danger"]["noul"]),
            urgency=float(answers["urgency"]["score"]),
            inference_ms=inference_ms,
            shielded=False,
            prompt=prompt,
        )

    ok = admissible(game, strategy)
    best = geometric_policy(game, strategy)
    prompt = describe(game, ok, best, strategy)
    started = time.perf_counter()
    result = agent.predict(prompt, build_questions(ok, best))
    inference_ms = (time.perf_counter() - started) * 1000.0
    answers = result["answers"]

    action_answer = answers["action"]
    raw_action = action_answer["choice"]
    probabilities = action_answer["probabilities"]
    danger = float(answers["danger"]["noul"])
    urgency = float(answers["urgency"]["score"])

    action = raw_action
    shielded = False
    if shield and raw_action not in ok and ok:
        action = "jump" if "jump" in ok else "glide"
        shielded = True
        # empty admissible set: nothing survives, execute the model's pick honestly

    return Decision(
        action=action,
        raw_action=raw_action,
        probabilities=probabilities,
        danger=danger,
        urgency=urgency,
        inference_ms=inference_ms,
        shielded=shielded,
        prompt=prompt,
    )
