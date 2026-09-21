"""Episode runner: NES emulator loop with pipelined local inference.

Kept from typesafe-mario — the inference thread pool, per-decision artifact
logging, and the realtime dashboard loop. The only change: decision records
also store the exact Laya prompt text.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .actions import ACTION_TO_INDEX, JUMP_ACTIONS, JUMP_RELEASE_ACTION, Action
from .dashboard import DashboardCommand, LiveDashboard
from .policy import Decision, Policy
from .state import MarioStateParser


def _unwrap_ram(env: Any) -> Any:
    current = env
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        ram = getattr(current, "ram", None)
        if ram is not None:
            return ram
        next_env = getattr(current, "env", None)
        if next_env is None:
            break
        current = next_env
    return None


def create_mario_env(env_id: str, render_mode: str = "human") -> Any:
    try:
        import gym_super_mario_bros  # noqa: F401
        import gymnasium as gym
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace
    except ImportError as exc:
        raise RuntimeError(
            'Mario dependencies are missing. Install with: uv sync --extra mario'
        ) from exc

    env = gym.make(env_id, render_mode=render_mode)
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def _record_decision(
    log: Any,
    *,
    decision_index: int,
    snapshot: Any,
    decision: Decision,
    reward: float,
    terminated: bool,
    truncated: bool,
    flag_get: bool = False,
) -> None:
    state = snapshot.to_state()
    if flag_get:
        # the emulator terminates ON flagpole contact; the snapshot was parsed
        # one decision earlier, so its RAM never saw the stage-clear flag.
        # Without this override every clear reads as an unexplained stop
        # (measured: six "died at the final staircase" runs were all clears)
        state["episode"] = {**state["episode"], "stage_clear": True}
    record = {
        "decision": decision_index,
        "state": state,
        "debug_state": snapshot.to_debug_state(),
        "state_text": snapshot.to_text(),
        "prompt": decision.prompt,
        "labels": dict(decision.labels) if decision.labels else None,
        "shielded": decision.shielded,
        "rescue": decision.rescue,
        "action": decision.action.value,
        "confidence": decision.confidence,
        "probabilities": dict(decision.probabilities),
        "jump_needed_probability": decision.jump_needed_probability,
        "danger_score": decision.danger_score,
        "latency_ms": decision.latency_ms,
        "reward": reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "flag_get": bool(flag_get),
    }
    log.write(json.dumps(record, separators=(",", ":")) + "\n")
    log.flush()


def _run_realtime_dashboard(
    *,
    env: Any,
    env_id: str,
    dashboard: LiveDashboard,
    policy: Policy,
    parser: MarioStateParser,
    frame: Any,
    info: dict[str, Any],
    log: Any,
    frames_per_decision: int,
    max_decisions: int,
    screenshot_path: Path | None,
    actions: tuple[Action, ...] = tuple(Action),
) -> Any:
    """Realtime dashboard with HEADLESS-IDENTICAL decision semantics.

    The original pipelined loop (decide while the emulator runs, apply on
    completion) produced a different trajectory from the measured headless
    runs — inference landed 1-2 frames late and the release-edge handling
    differed, re-rolling every knife-edge phase (the user watched the record
    config die at the first pit in the window while it reaches 93% headless).
    This loop is synchronous like headless: at each decision boundary the
    emulator pauses for one inference (~17ms, one dropped frame every
    frames_per_decision), the snapshot is parsed ONLY at boundaries with
    frames_elapsed=frames_per_decision, and the jump release edge inserts an
    extra frame exactly like the headless loop — so the window shows the
    same deterministic trajectory the jsonl record was made on.
    """
    actions = tuple(actions)
    active_decision: Decision | None = None
    next_index = 0
    episode_reward = 0.0
    previous_action: Action | None = None
    previous_reward = 0.0
    previous_latency_ms = 0.0
    screenshot_saved = False
    terminated = truncated = False
    shields = rescues = 0
    quit_requested = False

    level_commands = {
        DashboardCommand.LEVEL_1_1: "1-1",
        DashboardCommand.LEVEL_1_2: "1-2",
        DashboardCommand.LEVEL_1_3: "1-3",
        DashboardCommand.LEVEL_1_4: "1-4",
    }

    def _reset(stage: str | None) -> None:
        """Restart in place, or switch stage (1-1..1-4) with a fresh env."""
        nonlocal env, frame, info, active_decision, next_index, episode_reward
        nonlocal previous_action, previous_reward, previous_latency_ms
        nonlocal terminated, truncated, shields, rescues
        if stage is not None:
            env.close()
            env = create_mario_env(f"SuperMarioBros-{stage}-v0", render_mode="rgb_array")
        frame, info = env.reset()
        parser.reset()
        active_decision = None
        next_index = 0
        episode_reward = 0.0
        previous_action = None
        previous_reward = 0.0
        previous_latency_ms = 0.0
        terminated = truncated = False
        shields = rescues = 0
        print(f"--- {'Stage ' + stage if stage else 'Restarted'} ---")

    while not quit_requested:
        snapshot = parser.parse(
            info,
            _unwrap_ram(env),
            previous_action=previous_action.value if previous_action else None,
            previous_reward=previous_reward,
            previous_latency_ms=previous_latency_ms,
            previous_response_delay_frames=0,
            frames_elapsed=frames_per_decision,
        )
        run_ended = bool(
            snapshot.dead or snapshot.clear or terminated or truncated or next_index >= max_decisions
        )
        # flag_get arrives in the info dict of the terminating step; the parse
        # above happened one decision earlier and cannot see it
        cleared = bool(info.get("flag_get")) and (terminated or snapshot.clear)
        if not run_ended:
            decision = policy.choose(snapshot, actions)  # blocking, ~one frame
            total_reward = 0.0
            print(
                f"#{next_index:04d} x={snapshot.x:04d} "
                f"action={decision.action.value:<15} "
                f"confidence={decision.confidence:.2f} "
                f"latency={decision.latency_ms:.0f}ms"
                + (" [SHIELD]" if decision.shielded and not decision.rescue else "")
                + (" [RESCUE]" if decision.rescue else "")
            )
            shields += int(decision.shielded)
            rescues += int(decision.rescue)
            active_decision = decision
            previous_latency_ms = decision.latency_ms

            macro: list[Action] = []
            if (
                decision.action in JUMP_ACTIONS
                and previous_action in JUMP_ACTIONS
                and snapshot.grounded
            ):
                macro.append(JUMP_RELEASE_ACTION[decision.action])  # extra frame, like headless
            macro.extend(decision.action for _ in range(frames_per_decision))

            terminated = truncated = False
            for step_action in macro:
                frame, reward, terminated, truncated, info = env.step(
                    ACTION_TO_INDEX[step_action]
                )
                total_reward += float(reward)
                episode_reward += float(reward)
                command = dashboard.draw(
                    frame,
                    snapshot,
                    active_decision,
                    decision_index=next_index,
                    episode_reward=episode_reward,
                    waiting=False,
                    run_ended=False,
                    shield_count=shields,
                    rescue_count=rescues,
                )
                if command == DashboardCommand.QUIT:
                    quit_requested = True
                    break
                if command == DashboardCommand.RESTART:
                    break
                if terminated or truncated:
                    break
            _record_decision(
                log,
                decision_index=next_index,
                snapshot=snapshot,
                decision=decision,
                reward=total_reward,
                terminated=terminated,
                truncated=truncated,
                flag_get=bool(info.get("flag_get")),
            )
            previous_action = decision.action
            previous_reward = total_reward
            next_index += 1
            if not quit_requested and command in (
                DashboardCommand.RESTART,
                *level_commands,
            ):
                _reset(level_commands.get(command))
                continue

        if screenshot_path is not None and active_decision is not None and not screenshot_saved:
            dashboard.save(screenshot_path)
            screenshot_saved = True
        if not quit_requested:
            command = dashboard.draw(
                frame,
                snapshot,
                active_decision,
                decision_index=max(0, next_index - 1),
                episode_reward=episode_reward,
                waiting=False,
                run_ended=run_ended,
                stage_clear=cleared,
                shield_count=shields,
                rescue_count=rescues,
            )
            if command == DashboardCommand.QUIT:
                break
            if command in (DashboardCommand.RESTART, *level_commands):
                _reset(level_commands.get(command))
    return env


def run_episode(
    *,
    env_id: str,
    policy: Policy,
    frames_per_decision: int,
    max_decisions: int,
    seed: int,
    artifacts_dir: Path,
    display: str = "dashboard",
    screenshot_path: Path | None = None,
    actions: Sequence[Action] | None = None,
) -> Path:
    if frames_per_decision < 1:
        raise ValueError("frames_per_decision must be at least 1")

    if display not in {"dashboard", "game", "none"}:
        raise ValueError("display must be dashboard, game, or none")
    render_mode = "human" if display == "game" else "rgb_array"
    env = create_mario_env(env_id, render_mode=render_mode)
    dashboard = LiveDashboard() if display == "dashboard" else None
    parser = MarioStateParser(decision_horizon_frames=frames_per_decision)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = artifacts_dir / f"run-{timestamp}.jsonl"

    frame, info = env.reset(seed=seed)

    try:
        with log_path.open("w", encoding="utf-8") as log:
            if dashboard:
                env = (
                    _run_realtime_dashboard(
                        env=env,
                        env_id=env_id,
                        dashboard=dashboard,
                        policy=policy,
                        parser=parser,
                        frame=frame,
                        info=info,
                        log=log,
                        frames_per_decision=frames_per_decision,
                        max_decisions=max_decisions,
                        screenshot_path=screenshot_path,
                        actions=tuple(actions) if actions else tuple(Action),
                    )
                    or env
                )
                return log_path

            previous_action: Action | None = None
            previous_reward = 0.0
            previous_latency_ms = 0.0
            actions = tuple(actions) if actions else tuple(Action)
            shields = rescues = 0
            for decision_index in range(max_decisions):
                snapshot = parser.parse(
                    info,
                    _unwrap_ram(env),
                    previous_action=previous_action.value if previous_action else None,
                    previous_reward=previous_reward,
                    previous_latency_ms=previous_latency_ms,
                    previous_response_delay_frames=0,
                    frames_elapsed=frames_per_decision,
                )
                if snapshot.dead or snapshot.clear:
                    break

                decision = policy.choose(snapshot, actions)
                shields += int(decision.shielded)
                rescues += int(decision.rescue)
                action_index = ACTION_TO_INDEX[decision.action]
                total_reward = 0.0
                terminated = truncated = False
                # A jump macro needs a button-up edge before A counts again:
                # if A is still held from the previous macro (it was also a
                # jump) and Mario is grounded, spend one frame releasing first.
                # The dashboard loop already did this; the headless loop did
                # not, so Mario jumped exactly once and then stood pressed
                # against pipes.
                if (
                    decision.action in JUMP_ACTIONS
                    and previous_action in JUMP_ACTIONS
                    and snapshot.grounded
                ):
                    release = ACTION_TO_INDEX[JUMP_RELEASE_ACTION[decision.action]]
                    frame, reward, terminated, truncated, info = env.step(release)
                    total_reward += float(reward)
                    if terminated or truncated:
                        # the episode can end on the release frame itself
                        # (measured crash: spring f7 died exactly there)
                        _record_decision(
                            log,
                            decision_index=decision_index,
                            snapshot=snapshot,
                            decision=decision,
                            reward=total_reward,
                            terminated=terminated,
                            truncated=truncated,
                            flag_get=bool(info.get("flag_get")),
                        )
                        break
                for _ in range(frames_per_decision):
                    frame, reward, terminated, truncated, info = env.step(action_index)
                    total_reward += float(reward)
                    if terminated or truncated:
                        break

                _record_decision(
                    log,
                    decision_index=decision_index,
                    snapshot=snapshot,
                    decision=decision,
                    reward=total_reward,
                    terminated=terminated,
                    truncated=truncated,
                    flag_get=bool(info.get("flag_get")),
                )
                print(
                    f"#{decision_index:04d} x={snapshot.x:04d} "
                    f"action={decision.action.value:<15} "
                    f"confidence={decision.confidence:.2f} latency={decision.latency_ms:.0f}ms"
                    + (" [SHIELD]" if decision.shielded and not decision.rescue else "")
                    + (" [RESCUE]" if decision.rescue else "")
                    + (" [STAGE CLEAR]" if info.get("flag_get") else "")
                )

                previous_action = decision.action
                previous_reward = total_reward
                previous_latency_ms = decision.latency_ms
                if terminated or truncated:
                    break
    finally:
        if dashboard:
            dashboard.close()
        env.close()
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    return log_path
