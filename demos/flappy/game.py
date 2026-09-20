"""Discrete grid Flappy kernel. Pure logic: no rendering, no model, no I/O."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

ROWS = 24  # playable rows; the last row is drawn as ground
COLS = 48  # canvas width in cells
BIRD_X = 10  # fixed column of the bird
GAP = 8  # gap height in rows
PIPE_W = 3  # pipe width in columns
SPAWN_DIST = 14  # columns between pipe pairs
GRAVITY = 1  # rows gained downward per step
FLAP = -2  # vertical velocity set by a flap
MAX_ABS_VY = 2


@dataclass
class Pipe:
    x: int
    gap_center: int
    scored: bool = False

    @property
    def gap_top(self) -> int:
        return self.gap_center - GAP // 2

    @property
    def gap_bottom(self) -> int:  # exclusive
        return self.gap_top + GAP


@dataclass
class FlappyGame:
    seed: int = 0
    rows: int = ROWS
    cols: int = COLS
    bird_y: int = ROWS // 2 - 2
    vy: int = 0
    score: int = 0
    steps: int = 0
    alive: bool = True
    death_reason: str = ""
    pipes: list[Pipe] = field(default_factory=list)
    _rng: random.Random = field(default_factory=lambda: random.Random(0), repr=False, compare=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.pipes = [Pipe(self.cols + 4, self._rand_gap_center())]

    def clone(self) -> "FlappyGame":
        """Deep copy including RNG state, for what-if roll-outs."""
        other = FlappyGame.__new__(FlappyGame)
        other.__dict__.update(
            {
                "seed": self.seed,
                "rows": self.rows,
                "cols": self.cols,
                "bird_y": self.bird_y,
                "vy": self.vy,
                "score": self.score,
                "steps": self.steps,
                "alive": self.alive,
                "death_reason": self.death_reason,
                "pipes": [Pipe(p.x, p.gap_center, p.scored) for p in self.pipes],
                "_rng": random.Random(self.seed),
            }
        )
        other._rng.setstate(self._rng.getstate())
        return other

    def _rand_gap_center(self) -> int:
        return self._rng.randint(GAP // 2 + 1, self.rows - GAP // 2 - 2)

    # ------------------------------------------------------------------ state

    @property
    def next_pipe(self) -> Pipe | None:
        """First pipe whose right edge has not passed the bird."""
        for pipe in self.pipes:
            if pipe.x + PIPE_W - 1 >= BIRD_X:
                return pipe
        return None

    def columns_to_gap(self) -> int:
        """Columns from the bird to the right edge of the next gap (>=0)."""
        pipe = self.next_pipe
        if pipe is None:
            return 99
        return max(0, pipe.x + PIPE_W - 1 - BIRD_X)

    def collides(self, pipe: Pipe) -> bool:
        if not (pipe.x <= BIRD_X < pipe.x + PIPE_W):
            return False
        return self.bird_y <= pipe.gap_top or self.bird_y >= pipe.gap_bottom

    # ------------------------------------------------------------------ input

    def flap(self) -> None:
        if self.alive:
            self.vy = FLAP

    # ------------------------------------------------------------------- step

    def tick(self) -> None:
        if not self.alive:
            return
        self.steps += 1
        self.vy = max(-MAX_ABS_VY, min(MAX_ABS_VY, self.vy + GRAVITY))
        self.bird_y += self.vy

        if self.bird_y < 0:
            self._die("飞出顶端")
            return
        if self.bird_y > self.rows - 2:
            self._die("撞到地面")
            return

        last = self.pipes[-1]
        if self.cols - last.x >= SPAWN_DIST:
            self.pipes.append(Pipe(self.cols + 2, self._rand_gap_center()))

        for pipe in self.pipes:
            pipe.x -= 1
            if not pipe.scored and pipe.x + PIPE_W - 1 < BIRD_X:
                pipe.scored = True
                self.score += 1
        if self.pipes and self.pipes[0].x + PIPE_W < 0:
            self.pipes.pop(0)

        for pipe in self.pipes:
            if self.collides(pipe):
                self._die("撞上管道")
                return

    def _die(self, reason: str) -> None:
        self.alive = False
        self.death_reason = reason
