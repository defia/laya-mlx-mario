"""Single-state A/B probe: replay recorded snapshots against prompt variants.

Loads a headless run log (jsonl), rebuilds each MarioSnapshot, and asks the
local model the same typed questions under different prompt configurations
(background on/off, verdicts on/off). Used to measure how a wording change
shifts the choice distribution at the exact states that killed Mario —
without burning whole episodes on noise (single-run A/B is noise; see README
research note 2).

Usage:
  uv run python -m demos.mario.probe <run.jsonl> [--decisions 12,13] \
      [--where enemy|gap|wall] [--limit 8] [--model ...]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .actions import Action
from .describe import build_questions, describe
from .state import EnemyObservation, MarioSnapshot

DEFAULT_MODEL = "models/hub/laya-multilingual-mlx"


def _stomp_note(prompt: str) -> str:
    """Restate the generic stomp rule inside the enemy-fact sentence."""
    return prompt.replace(
        "帧后接触。", "帧后接触，从上方落到它头顶可以踩扁它。"
    )


def _goal_instructions(questions: dict) -> dict:
    questions = json.loads(json.dumps(questions))  # deep copy
    questions["next_action"]["instructions"] = "选择能让马里奥活着向右前进的下一个手柄动作。"
    return questions


def _restrict_actions(keep: tuple[str, ...]):
    """Rebuild the choice criteria over a smaller action subset."""

    def transform(questions: dict, actions=()) -> dict:
        from .actions import Action
        from .describe import BASE_LABELS

        questions = json.loads(json.dumps(questions))
        questions["next_action"]["criteria"] = {
            name: BASE_LABELS[Action(name)] for name in keep
        }
        return questions

    return transform


def _cadence_note(prompt: str) -> str:
    """Tell the model one action lasts 8 frames (game-background fact)."""
    return prompt.replace("原地约3格。", "原地约3格。一次手柄动作持续8帧。")


VARIANTS = {
    "pure": {"background": True, "verdicts": False},
    "facts-only": {"background": False, "verdicts": False},
    "assist": {"background": False, "verdicts": True},
    "stomp": {
        "background": True,
        "verdicts": False,
        "prompt_transform": _stomp_note,
    },
    "goal-instr": {
        "background": True,
        "verdicts": False,
        "question_transform": _goal_instructions,
    },
    "a5": {
        "background": True,
        "verdicts": False,
        "question_transform": _restrict_actions(
            ("right", "right_jump", "right_run", "right_run_jump", "left")
        ),
    },
    "cadence": {
        "background": True,
        "verdicts": False,
        "prompt_transform": _cadence_note,
    },
    "a5+cadence": {
        "background": True,
        "verdicts": False,
        "question_transform": _restrict_actions(
            ("right", "right_jump", "right_run", "right_run_jump", "left")
        ),
        "prompt_transform": _cadence_note,
    },
}


def snapshot_from_record(record: dict) -> MarioSnapshot:
    """Rebuild the MarioSnapshot a jsonl decision was made on."""
    state = record["state"]
    debug = record["debug_state"]
    level = state["level"]
    player = state["player"]
    trajectory = state["trajectory"]
    control = state["recent_control"]
    episode = state["episode"]
    raw = debug["raw"]
    enemies = tuple(
        EnemyObservation(
            slot=e["slot"],
            kind_id=e["kind_id"],
            kind=e["kind"],
            dx_pixels=e["relative_x_pixels"],
            dy_pixels=e["relative_y_pixels"],
            relative_velocity_x=e["relative_velocity_x"],
        )
        for e in debug["visible_enemies"]
    )
    return MarioSnapshot(
        goal=state["objective"],
        world=level["world"],
        stage=level["stage"],
        area=level["area"],
        x=player["x"],
        y=player["y"],
        dx=player["horizontal_speed_px_per_frame"],
        dy=player["vertical_speed_px_per_frame"],
        direction=raw["horizontal_motion"],
        vertical_motion=raw["vertical_motion"],
        airborne=not player["grounded"],
        status=player["powerup_status"],
        player_state=raw["player_state"],
        lives=episode["lives"],
        coins=raw["coins"],
        score=raw["score"],
        time_left=episode["time_left"],
        progress=episode["progress"],
        best_progress=episode["best_progress"],
        stalled_steps=episode["stalled_frames"],
        enemies=enemies,
        local_grid=tuple(debug["local_grid"]["rows"]),
        previous_action=control["action"],
        previous_reward=raw["previous_reward"],
        dead=episode["dead"],
        clear=episode["stage_clear"],
        grounded=player["grounded"],
        jump_phase=player["jump_phase"],
        action_frames=control["frames_observed"],
        action_progress=control["progress_gained_pixels"],
        airborne_frames=trajectory["airborne_frames"],
        jump_distance_pixels=trajectory["horizontal_distance_since_takeoff_pixels"],
        crossing_gap=trajectory["crossing_known_gap"],
        gap_width_at_commit=trajectory["gap_width_at_commit_tiles"],
        decision_horizon_frames=state["reaction_timing"]["action_horizon_frames"],
        last_response_delay_frames=state["reaction_timing"]["last_inference_delay_frames"],
    )


def _matches(snapshot: MarioSnapshot, where: str) -> bool:
    if where == "all":
        return True
    hazard = snapshot.threat_features()
    if where == "enemy":
        return bool(hazard.get("enemy_ahead"))
    if where == "enemy-close":
        contact = hazard.get("estimated_contact_frames")
        return bool(hazard.get("enemy_ahead")) and contact is not None and contact <= 40
    if where == "gap":
        terrain = snapshot.navigation_features()
        d = terrain.get("gap_distance_tiles")
        return terrain.get("geometry_available") and d is not None and d <= 5
    if where == "wall":
        terrain = snapshot.navigation_features()
        return (
            snapshot.grounded
            and terrain.get("geometry_available")
            and terrain.get("obstacle_distance_tiles") is not None
            and (terrain.get("obstacle_height_tiles") or 0) >= 3
        )
    raise ValueError(f"unknown --where {where!r}")


def run_probe(
    log_path: Path,
    *,
    decisions: Sequence[int] | None = None,
    where: str = "all",
    limit: int = 6,
    model: str = DEFAULT_MODEL,
    variants: dict[str, dict] | None = None,
) -> None:
    from laya_mlx import Agent

    agent = Agent(model, dtype="float16")
    variants = variants or VARIANTS
    actions = tuple(Action)
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    picked = []
    for record in records:
        if decisions and record["decision"] not in decisions:
            continue
        snapshot = snapshot_from_record(record)
        if not _matches(snapshot, where):
            continue
        picked.append((record, snapshot))
        if len(picked) >= limit:
            break

    jump_family = ("right_jump", "jump", "right_run_jump")
    for record, snapshot in picked:
        contact = snapshot.threat_features().get("estimated_contact_frames")
        print(
            f"\n=== #{record['decision']:04d} x={snapshot.x} dx={snapshot.dx} "
            f"grounded={snapshot.grounded} contact={contact} "
            f"actual={record['action']} ==="
        )
        for name, cfg in variants.items():
            prompt = describe(snapshot, "zh", background=cfg["background"])
            transform = cfg.get("prompt_transform")
            if transform is not None:
                prompt = transform(prompt)
            questions = build_questions(
                snapshot, actions, "zh", verdicts=cfg["verdicts"]
            )
            question_transform = cfg.get("question_transform")
            if question_transform is not None:
                questions = question_transform(questions)
            result = agent.predict(prompt, questions)
            answer = result["answers"]["next_action"]
            probs = {k: float(v) for k, v in dict(answer["probabilities"]).items()}
            top = max(probs, key=probs.get)
            jump_mass = sum(probs.get(a, 0.0) for a in jump_family)
            print(
                f"  {name:<10} top={top:<15} jump家族={jump_mass:.2f} "
                + " ".join(f"{a}={probs.get(a, 0):.2f}" for a in
                           ("right", "right_jump", "right_run", "right_run_jump", "jump", "left", "noop"))
            )
    close = getattr(agent, "close", None)
    if callable(close):
        close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="laya-mario-probe")
    parser.add_argument("log", type=Path)
    parser.add_argument("--decisions", default="", help="comma-separated decision indices")
    parser.add_argument("--where", default="all", help="all|enemy|enemy-close|gap|wall")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    decisions = (
        {int(d) for d in args.decisions.split(",") if d.strip()} or None
    )
    run_probe(
        args.log,
        decisions=decisions,
        where=args.where,
        limit=args.limit,
        model=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
