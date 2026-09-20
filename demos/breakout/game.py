"""Discrete grid Breakout kernel. Pure logic: no rendering, no model, no I/O.

The ball always moves one cell per tick along a diagonal (vx, vy ∈ {-1, +1}).
Bricks bounce it straight back. The paddle acts like a curved face: a center
hit mirrors the ball back where it came from, while hitting the left/right
edge forces the outbound direction to that side — which is what makes paddle
positioning a real decision.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

ROWS = 24
COLS = 42
BRICK_TOP = 2  # first brick row
BRICK_ROWS = 6  # wall height in rows
BRICK_W = 2  # each brick spans 2 columns: cell (y, x) hits brick (y, x // 2)
PADDLE_W = 7
PADDLE_ROW = ROWS - 3
PADDLE_SPEED = 2  # columns per move; the ball covers 1 per tick
LIVES = 3
SERVE_ROW = ROWS // 2


@dataclass
class BreakoutGame:
    rows: int = ROWS
    cols: int = COLS
    ball_x: int = COLS // 2
    ball_y: int = SERVE_ROW
    vx: int = 1
    vy: int = -1
    px: int = (COLS - PADDLE_W) // 2
    bricks: set[tuple[int, int]] | None = None  # None = build a fresh wall
    lives: int = LIVES
    score: int = 0  # combo-weighted points; the k-th brick of a rally is worth k
    combo: int = 0  # bricks broken in the current rally (since the last paddle hit)
    best_combo: int = 0
    broken: int = 0  # total bricks broken (count)
    level: int = 1
    steps: int = 0
    alive: bool = True
    death_reason: str = ""
    last_event: str = ""  # "" | wall | brick | paddle | miss | clear
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if self.bricks is None:
            self.bricks = self._wall()
            self.serve()

    def clone(self) -> "BreakoutGame":
        """Deep copy including RNG state, for what-if roll-outs."""
        other = BreakoutGame.__new__(BreakoutGame)
        other.__dict__.update(
            {
                "rows": self.rows,
                "cols": self.cols,
                "ball_x": self.ball_x,
                "ball_y": self.ball_y,
                "vx": self.vx,
                "vy": self.vy,
                "px": self.px,
                "bricks": set(self.bricks or ()),
                "lives": self.lives,
                "score": self.score,
                "combo": self.combo,
                "best_combo": self.best_combo,
                "broken": self.broken,
                "level": self.level,
                "steps": self.steps,
                "alive": self.alive,
                "death_reason": self.death_reason,
                "last_event": self.last_event,
                "seed": self.seed,
                "_rng": random.Random(self.seed),
            }
        )
        other._rng.setstate(self._rng.getstate())
        return other

    def _wall(self) -> set[tuple[int, int]]:
        # Top two rows are solid; below them, channels run along the up-right
        # diagonal (b + r = const), one every 5th brick. Only an up-right ball
        # can ride a channel deep into the wall, which makes the receive-point
        # choice (strategy "steer") genuinely decide combo rallies.
        return {
            (r, b)
            for r in range(BRICK_TOP, BRICK_TOP + BRICK_ROWS)
            for b in range(self.cols // BRICK_W)
            if r < BRICK_TOP + 2 or (b + r) % 5
        }

    def serve(self) -> None:
        self.ball_x = self.cols // 2 + self._rng.randint(-6, 6)
        self.ball_y = SERVE_ROW
        self.vx = self._rng.choice((-1, 1))
        self.vy = -1
        self.combo = 0

    # ------------------------------------------------------------------ state

    @property
    def paddle_center(self) -> int:
        return self.px + PADDLE_W // 2

    def landing(self, max_steps: int = 320) -> tuple[int | None, int]:
        """(column, steps) where the ball next crosses the paddle row going
        down, computed with the paddle removed from play. (None, max_steps)
        if it does not come down within the horizon."""
        scratch = self.clone()
        scratch.last_event = ""
        for step in range(1, max_steps + 1):
            descending = scratch.vy > 0
            scratch.tick(ignore_paddle=True)
            if descending and scratch.ball_y >= PADDLE_ROW:
                return scratch.ball_x, step
        return None, max_steps

    # ------------------------------------------------------------------ input

    def move(self, action: str) -> None:
        if action == "left":
            self.px = max(0, self.px - PADDLE_SPEED)
        elif action == "right":
            self.px = min(self.cols - PADDLE_W, self.px + PADDLE_SPEED)

    # ------------------------------------------------------------------- step

    def tick(self, ignore_paddle: bool = False) -> None:
        if not self.alive:
            return
        self.steps += 1
        self.last_event = ""
        nbx = self.ball_x + self.vx
        nby = self.ball_y + self.vy

        if nbx < 0 or nbx >= self.cols:
            self.vx = -self.vx
            nbx = self.ball_x + self.vx
            self.last_event = "wall"
        if nby < 0:
            self.vy = -self.vy
            nby = self.ball_y + self.vy
            self.last_event = "wall"
        if (nby, nbx // BRICK_W) in self.bricks:
            self.bricks.discard((nby, nbx // BRICK_W))
            self.combo += 1
            self.best_combo = max(self.best_combo, self.combo)
            self.broken += 1
            self.score += self.combo  # rally bonus: the k-th brick of a rally scores k
            self.vy = -self.vy
            nby = self.ball_y + self.vy
            self.last_event = "brick"
        if (
            not ignore_paddle
            and self.vy > 0
            and nby == PADDLE_ROW
            and self.px <= nbx < self.px + PADDLE_W
        ):
            self.vy = -self.vy
            offset = nbx - self.paddle_center
            if offset <= -2:  # edge hits steer the outbound direction
                self.vx = -1
            elif offset >= 2:
                self.vx = 1
            else:
                self.vx = -self.vx  # curved face: a center hit mirrors the ball
            # depart with the NEW direction in the same tick, so the ball
            # leaves cleanly instead of sliding one step the old way first
            nbx = self.ball_x + self.vx
            nby = self.ball_y + self.vy
            self.last_event = "paddle"
            self.combo = 0

        self.ball_x, self.ball_y = nbx, nby

        if nby >= self.rows:
            self.lives -= 1
            self.last_event = "miss"
            if self.lives <= 0:
                self.alive = False
                self.death_reason = "三球全漏"
            else:
                self.serve()
        elif not self.bricks:
            self.level += 1
            self.bricks = self._wall()
            self.serve()
            self.last_event = "clear"
