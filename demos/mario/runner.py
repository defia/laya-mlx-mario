"""Episode runner: NES emulator loop with pipelined local inference.

Kept from typesafe-mario — the inference thread pool, per-decision artifact
logging, and the realtime dashboard loop. The only change: decision records
also store the exact Laya prompt text.
"""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
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
) -> None:
    record = {
        "decision": decision_index,
        "state": snapshot.to_state(),
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
    }
    log.write(json.dumps(record, separators=(",", ":")) + "\n")
    log.flush()


def _run_realtime_dashboard(
    *,
    env: Any,
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
) -> None:
    actions = tuple(actions)
    active_decision: Decision | None = None
    pending: Future[Decision] | None = None
    pending_snapshot: Any = None
    pending_index = 0
    pending_request_frame = 0
    next_index = 0
    frame_index = 0
    last_request_frame = -frames_per_decision
    episode_reward = 0.0
    reward_since_decision = 0.0
    previous_action: Action | None = None
    previous_reward = 0.0
    previous_latency_ms = 0.0
    previous_response_delay_frames = 0
    screenshot_saved = False
    terminated = truncated = False
    shields = rescues = 0

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-mario") as executor:
        while True:
            decision_updated = False
            snapshot = parser.parse(
                info,
                _unwrap_ram(env),
                previous_action=previous_action.value if previous_action else None,
                previous_reward=previous_reward,
                previous_latency_ms=previous_latency_ms,
                previous_response_delay_frames=previous_response_delay_frames,
            )

            if pending is not None and pending.done():
                active_decision = pending.result()
                _record_decision(
                    log,
                    decision_index=pending_index,
                    snapshot=pending_snapshot,
                    decision=active_decision,
                    reward=reward_since_decision,
                    terminated=terminated,
                    truncated=truncated,
                )
                print(
                    f"#{pending_index:04d} x={pending_snapshot.x:04d} "
                    f"action={active_decision.action.value:<15} "
                    f"confidence={active_decision.confidence:.2f} "
                    f"latency={active_decision.latency_ms:.0f}ms"
                    + (" [SHIELD]" if active_decision.shielded and not active_decision.rescue else "")
                    + (" [RESCUE]" if active_decision.rescue else "")
                )
                shields += int(active_decision.shielded)
                rescues += int(active_decision.rescue)
                pending = None
                reward_since_decision = 0.0
                decision_updated = True
                previous_latency_ms = active_decision.latency_ms
                previous_response_delay_frames = frame_index - pending_request_frame

            run_ended = bool(
                snapshot.dead
                or snapshot.clear
                or terminated
                or truncated
                or (next_index >= max_decisions and pending is None)
            )

            if (
                not run_ended
                and pending is None
                and next_index < max_decisions
                and frame_index - last_request_frame >= frames_per_decision
            ):
                pending_snapshot = snapshot
                pending_index = next_index
                pending = executor.submit(policy.choose, snapshot, actions)
                pending_request_frame = frame_index
                next_index += 1
                last_request_frame = frame_index

            if not run_ended:
                action = active_decision.action if active_decision else Action.NOOP
                if decision_updated and action in JUMP_ACTIONS and snapshot.grounded:
                    # A new jump macro needs a button-up edge before A is pressed again.
                    action = JUMP_RELEASE_ACTION[action]
                frame, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
                previous_action = action
                previous_reward = float(reward)
                reward_since_decision += float(reward)
                episode_reward += float(reward)
                frame_index += 1

            command = dashboard.draw(
                frame,
                snapshot,
                active_decision,
                decision_index=max(0, next_index - 1),
                episode_reward=episode_reward,
                waiting=pending is not None,
                run_ended=run_ended,
                shield_count=shields,
                rescue_count=rescues,
            )
            if screenshot_path is not None and active_decision is not None and not screenshot_saved:
                dashboard.save(screenshot_path)
                screenshot_saved = True
            if command == DashboardCommand.QUIT:
                break
            if command == DashboardCommand.RESTART:
                if pending is not None:
                    pending.cancel()
                frame, info = env.reset()
                parser.reset()
                active_decision = None
                pending = None
                pending_snapshot = None
                pending_index = 0
                pending_request_frame = 0
                next_index = 0
                frame_index = 0
                last_request_frame = -frames_per_decision
                episode_reward = 0.0
                reward_since_decision = 0.0
                previous_action = None
                previous_reward = 0.0
                previous_latency_ms = 0.0
                previous_response_delay_frames = 0
                terminated = truncated = False
                shields = rescues = 0
                print("--- Restarted ---")


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
                _run_realtime_dashboard(
                    env=env,
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
                )
                print(
                    f"#{decision_index:04d} x={snapshot.x:04d} "
                    f"action={decision.action.value:<15} "
                    f"confidence={decision.confidence:.2f} latency={decision.latency_ms:.0f}ms"
                    + (" [SHIELD]" if decision.shielded and not decision.rescue else "")
                    + (" [RESCUE]" if decision.rescue else "")
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
