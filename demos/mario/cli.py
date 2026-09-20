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

DEFAULT_MODEL = "models/hub/laya-multilingual-mlx"


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
        default=8,
        help="Minimum macro duration in emulator frames",
    )
    play.add_argument("--max-decisions", type=int, default=2000)
    play.add_argument("--seed", type=int, default=123)
    play.add_argument("--policy", choices=("laya", "heuristic"), default="laya")
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
            policy = LayaPolicy(args.model, dtype=args.dtype, lang=lang)
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
        )
        print(f"Run log: {log_path.resolve()}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
