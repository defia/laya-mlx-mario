"""Entry point: uv run --extra demo python -m demos.breakout [options]"""

from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty

import laya_mlx as laya
from rich.console import Console

from .describe import STRATEGIES
from .game import BreakoutGame
from .player import decide
from .ui import live, render

DEFAULT_MODEL = "models/hub/laya-multilingual-mlx"


class KeyReader:
    """Single-key non-blocking reader in cbreak mode."""

    def __init__(self) -> None:
        self._old = None

    def __enter__(self) -> "KeyReader":
        if sys.stdin.isatty():
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc) -> None:
        if self._old is not None:
            termios.tcsetattr(sys.stdin, sys.stdin.fileno(), self._old)

    def poll(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="laya-breakout", description="Laya plays Breakout locally")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="local dir or Hub checkpoint id")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=15, help="decision steps per second")
    parser.add_argument("--unassisted", action="store_true", help="disable the drop shield")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        default="steady",
        help="receive strategy: steady=catch with the paddle center, steer=catch with an edge to aim",
    )
    parser.add_argument("--headless", action="store_true", help="no terminal display")
    parser.add_argument("--steps", type=int, default=300, help="headless decision steps")
    return parser.parse_args(argv)


class RunStats:
    def __init__(self) -> None:
        self.episodes = 0
        self.best = 0
        self.scores: list[int] = []
        self.bricks_total = 0
        self.best_combo = 0
        self.decisions = 0
        self.shields = 0
        self.misses = 0
        self.inference_ms: list[float] = []
        self._recent: list[float] = []

    def note_decision(self, inference_ms: float, shielded: bool) -> None:
        now = time.monotonic()
        self.decisions += 1
        self.shields += int(shielded)
        self.inference_ms.append(inference_ms)
        self._recent.append(now)
        while self._recent and now - self._recent[0] > 5.0:
            self._recent.pop(0)

    @property
    def rate(self) -> float:
        if len(self._recent) < 2:
            return 0.0
        span = self._recent[-1] - self._recent[0]
        return (len(self._recent) - 1) / span if span > 0 else 0.0

    def note_death(self, game: BreakoutGame) -> None:
        self.episodes += 1
        self.scores.append(game.score)
        self.bricks_total += game.broken
        self.best = max(self.best, game.score)
        self.best_combo = max(self.best_combo, game.best_combo)

    def view(self) -> dict:
        return {
            "episodes": self.episodes,
            "best": self.best,
            "decisions": self.decisions,
            "shields": self.shields,
            "rate": self.rate,
        }

    def summary(self, extra: dict | None = None) -> dict:
        times = sorted(self.inference_ms)
        out = {
            "episodes": self.episodes,
            "scores": self.scores,
            "best_score": self.best,
            "bricks": self.bricks_total,
            "best_combo": self.best_combo,
            "misses": self.misses,
            "decisions": self.decisions,
            "shields": self.shields,
            "mean_inference_ms": round(sum(times) / len(times), 2) if times else 0.0,
            "p50_inference_ms": round(times[len(times) // 2], 2) if times else 0.0,
            "p95_inference_ms": round(times[min(len(times) - 1, int(len(times) * 0.95))], 2) if times else 0.0,
        }
        if extra:
            out.update(extra)
        return out


def step_game(agent, game: BreakoutGame, stats: RunStats, shield: bool, strategy: str = "steady"):
    """One Laya decision + one physics tick. Returns the decision."""
    decision = decide(agent, game, shield=shield, strategy=strategy)
    stats.note_decision(decision.inference_ms, decision.shielded)
    game.move(decision.action)
    game.tick()
    if game.last_event == "miss":
        stats.misses += 1
    return decision


def run_headless(agent, args, stats: RunStats, shield: bool) -> None:
    started = time.monotonic()
    seed = args.seed
    while stats.decisions < args.steps:
        game = BreakoutGame(seed=seed)
        seed += 1
        while game.alive and stats.decisions < args.steps:
            step_game(agent, game, stats, shield, args.strategy)
        stats.note_death(game)
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            stats.summary(
                {
                    "steps_requested": args.steps,
                    "strategy": args.strategy,
                    "seconds": round(elapsed, 2),
                    "shield": shield,
                    "network": "offline",
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def run_live(agent, args, stats: RunStats, shield: bool) -> None:
    console = Console()
    frame_budget = 1.0 / max(1, args.fps)
    seed = args.seed
    strategy = args.strategy
    strategy_keys = {"1": "steady", "2": "steer"}
    log: list[dict] = []
    try:
        with live() as display, KeyReader() as keys:
            while True:
                game = BreakoutGame(seed=seed)
                seed += 1
                paused = False
                decision = None
                while game.alive:
                    frame_started = time.monotonic()
                    key = keys.poll()
                    if key and key.lower() == "q":
                        raise KeyboardInterrupt
                    if key and key.lower() == "p":
                        paused = not paused
                    if key in strategy_keys:
                        strategy = strategy_keys[key]
                    while paused:
                        time.sleep(0.05)
                        key = keys.poll()
                        if key and key.lower() == "q":
                            raise KeyboardInterrupt
                        if key and key.lower() == "p":
                            paused = False

                    decision = step_game(agent, game, stats, shield, strategy)
                    log.append(
                        {
                            "step": game.steps,
                            "prompt": decision.prompt,
                            "left_p": float(decision.probabilities.get("left", 0.0)),
                            "stay_p": float(decision.probabilities.get("stay", 0.0)),
                            "right_p": float(decision.probabilities.get("right", 0.0)),
                            "pick": decision.raw_action,
                            "exec": decision.action,
                            "shielded": decision.shielded,
                            "danger": decision.danger,
                            "urgency": decision.urgency,
                            "ms": decision.inference_ms,
                            "combo": game.combo,
                        }
                    )
                    if len(log) > 400:
                        del log[:200]
                    if game.last_event == "miss" and game.alive:
                        log[-1]["missed"] = game.lives
                    if not game.alive:
                        log[-1]["died"] = game.death_reason
                    view = stats.view()
                    view["strategy"] = STRATEGIES[strategy]["name"]
                    display.update(render(game, decision, view, log))
                    sleep_for = frame_budget - (time.monotonic() - frame_started)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                stats.note_death(game)
                view = stats.view()
                view["strategy"] = STRATEGIES[strategy]["name"]
                display.update(render(game, decision, view, log))
                time.sleep(0.9)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        console.print()
        console.print_json(
            json.dumps(
                stats.summary({"network": "offline", "shield": shield, "strategy": strategy}),
                ensure_ascii=False,
            )
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    shield = not args.unassisted
    console = Console()

    with console.status(f"[bold cyan]加载模型 {args.model} ..."):
        agent = laya.load(args.model, dtype="float16")
        # warm-up so the first real step is not skewed by lazy init
        step_game(agent, BreakoutGame(seed=args.seed), RunStats(), shield=False, strategy=args.strategy)

    stats = RunStats()
    if args.headless:
        run_headless(agent, args, stats, shield)
    else:
        run_live(agent, args, stats, shield)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
