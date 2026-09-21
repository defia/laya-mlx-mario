"""CLI: state-demo (no model, no emulator) and play (full episode)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .actions import Action
from .describe import build_questions, describe
from .policy import HeuristicPolicy, LayaPolicy
from .runner import run_episode
from .state import MarioStateParser

# Distance record config: the English 421M checkpoint under pure mode reaches
# x=2466 (73% of 1-1) with the step menu — see README. The multilingual
# checkpoint (zh) tops out at x=1763 with spring.
DEFAULT_MODEL = "models/hub/laya-mlx"

# The only instruction that creates mixed run/jump behavior without advice:
# a constant factual preference statement. English translations backfire
# ("Running is the fastest..." anchors right_run to 0.91) — keep it Chinese.
RUN_HINT = "平整的地面上向前奔跑是更快的移动方式。"

# Action menus. "core" drops noop and in-place jump: with 3 forward jumps in
# the menu the jump intent splits three ways and never wins argmax at the
# states that need it (measured: contact-16 goomba state, jump family 0.51 vs
# right 0.22 yet right wins because 0.51 is spread over three options).
ACTION_MENUS: dict[str, tuple[Action, ...]] = {
    "full": tuple(Action),
    "core": (
        Action.RIGHT,
        Action.RIGHT_JUMP,
        Action.RIGHT_RUN,
        Action.RIGHT_RUN_JUMP,
        Action.LEFT,
    ),
    # single forward speed: no plain "right" to split the advance intent,
    # so the two forward jumps win argmax at close-enemy states (measured:
    # right stays plurality 0.27 at every enemy distance otherwise)
    "hop": (
        Action.RIGHT_JUMP,
        Action.RIGHT_RUN,
        Action.RIGHT_RUN_JUMP,
        Action.LEFT,
    ),
    # walk-speed arcs only: short hops cannot overshoot the landing strip
    # after the stair pyramids (the hop menu's measured death at 1123/1412)
    "step": (
        Action.RIGHT_JUMP,
        Action.LEFT,
    ),
    # walk-jump + run-jump: lets the model build speed on flats (longer arcs
    # off the pyramid tops) without a plain-walk option diluting jump intent
    "spring": (
        Action.RIGHT_JUMP,
        Action.RIGHT_RUN_JUMP,
        Action.LEFT,
    ),
    # spring minus the never-chosen retreat (P(left) 0.03-0.10 everywhere)
    "spring2": (
        Action.RIGHT_JUMP,
        Action.RIGHT_RUN_JUMP,
    ),
    # the mixed gait: jump option FIRST (option order shifts probabilities
    # ~0.2 — see README note 12), paired with --run-hint it runs on flats and
    # jumps on the enemy/terrain triggers. Caps at the first stair pyramid
    # (x~845); behaviorally the most human-like pure-mode gait.
    "mix": (
        Action.RIGHT_JUMP,
        Action.RIGHT_RUN,
        Action.LEFT,
    ),
}


def _demo_ram() -> bytearray:
    ram = bytearray(0x0800)
    # Mario position and one goomba ahead.
    ram[0x006D] = 0
    ram[0x0086] = 172
    ram[0x000F] = 1
    ram[0x0016] = 0x06
    ram[0x006E] = 0
    ram[0x0087] = 214
    ram[0x00CF] = 79
    # Fill the tile-map row immediately below Mario with solid ground.
    ground_row = (79 + 16 - 32) // 16
    for column in range(16):
        ram[0x0500 + ground_row * 16 + column] = 1
    return ram


def state_demo() -> int:
    info = {
        "world": 1,
        "stage": 1,
        "area": 1,
        "x_pos": 172,
        "y_pos": 79,
        "y_pixel": 79,
        "left_x_pos": 60,
        "progress": 172,
        "progress_max": 172,
        "status": "small",
        "player_state": 8,
        "life": 2,
        "coins": 0,
        "score": 200,
        "time": 387,
        "death": False,
        "clear": False,
    }
    snapshot = MarioStateParser().parse(info, _demo_ram(), previous_action="right")
    print(json.dumps(snapshot.to_state(), indent=2, ensure_ascii=False))
    print("\n--- TEXT VIEW ---\n")
    print(snapshot.to_text())
    print("\n--- LAYA PROMPT (what the local model actually reads) ---\n")
    print(describe(snapshot))
    print("\n--- TYPED QUESTIONS ---\n")
    print(json.dumps(build_questions(snapshot, tuple(Action)), indent=2, ensure_ascii=False))
    print("\n--- ENGLISH PROMPT (for the English-only checkpoint) ---\n")
    print(describe(snapshot, "en"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laya-mario")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("state-demo", help="Show structured state and the Laya prompt")

    play = subparsers.add_parser("play", help="Run a Mario episode")
    play.add_argument("--env", default="SuperMarioBros-1-1-v0")
    play.add_argument(
        "--frames-per-decision",
        type=int,
        default=10,
        help="Decision cadence in emulator frames; 10 is the measured best (pure+step+EN)",
    )
    play.add_argument("--max-decisions", type=int, default=2000)
    play.add_argument("--seed", type=int, default=123)
    play.add_argument("--policy", choices=("laya", "heuristic"), default="laya")
    play.add_argument(
        "--mode",
        choices=("pure", "assist"),
        default="pure",
        help=(
            "pure: facts + game background only, no shield/rescue/verdict tags "
            "(the model judges alone); assist: shield + rescue + verdict tags"
        ),
    )
    play.add_argument(
        "--no-background",
        action="store_true",
        help="pure mode without the constant game-rules block (research A/B)",
    )
    play.add_argument(
        "--actions",
        choices=tuple(ACTION_MENUS),
        default="step",
        help=(
            "Controller menu: step = walk-speed arcs (pure-mode record), "
            "spring = walk-jump + run-jump (zh best), hop = 4 macros, "
            "mix = run + triggered jumps (most human-like, needs --run-hint), "
            "core = 5, full = 7"
        ),
    )
    play.add_argument(
        "--run-hint",
        action="store_true",
        help=(
            "Append the constant fact 'running on flat ground is faster' — with "
            "--actions mix this produces run-on-flats / jump-on-trigger behavior"
        ),
    )
    play.add_argument("--model", default=DEFAULT_MODEL, help="Local laya checkpoint directory")
    play.add_argument(
        "--lang",
        choices=("zh", "en"),
        default=None,
        help="Prompt language; default auto: multilingual checkpoint -> zh, else en",
    )
    play.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    play.add_argument(
        "--display",
        choices=("dashboard", "game", "none"),
        default="dashboard",
        help="Combined telemetry dashboard, plain game window, or headless mode",
    )
    play.add_argument("--artifacts-dir", default=Path(__file__).resolve().parent / "artifacts")
    play.add_argument(
        "--screenshot",
        type=Path,
        help="Save the first populated dashboard frame as a PNG",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "state-demo":
        return state_demo()
    if args.command == "play":
        if args.policy == "laya":
            lang = args.lang or ("zh" if "multilingual" in args.model else "en")
            policy = LayaPolicy(
                args.model,
                dtype=args.dtype,
                lang=lang,
                mode=args.mode,
                background=False if args.no_background else None,
                instruction=RUN_HINT if args.run_hint else "",
            )
        else:
            policy = HeuristicPolicy()
        log_path = run_episode(
            env_id=args.env,
            policy=policy,
            frames_per_decision=args.frames_per_decision,
            max_decisions=args.max_decisions,
            seed=args.seed,
            artifacts_dir=Path(args.artifacts_dir),
            display=args.display,
            screenshot_path=args.screenshot,
            actions=ACTION_MENUS[args.actions],
        )
        print(f"Run log: {log_path.resolve()}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
