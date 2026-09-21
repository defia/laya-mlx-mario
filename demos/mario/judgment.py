"""Judgment benchmark: does the model actually DECIDE at event states?

Distance scores conflate two things: the gait the menu induces (searched by
us) and the model's situational choices. This module isolates the second:

1. Harvest decision states from run logs (jsonl), rebuild snapshots, dedupe.
2. Bucket them by situation (enemy distance band, geometric gap verdict,
   no-event open field).
3. Ask the model ONE question per state with a pinned menu, and measure:
   - discrimination: does the jump-family probability MOVE when only the
     decision-relevant variable changes? (zero gradient == no deciding)
   - accuracy: choices vs parser-arithmetic ground truth (jump iff the gap
     verdict is "now"; jump iff enemy contact <= 24 frames; advance on open
     field), against random / always-jump / always-advance baselines.

Usage:
  uv run python -m demos.mario.judgment demos/mario/artifacts /tmp/mario-bgsearch
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

from .actions import Action
from .describe import gap_takeoff_verdict
from .probe import snapshot_from_record
from .state import MarioSnapshot

# pinned menus and orders: rj first (first-position law), order shifts
# probabilities ~0.2 (README note 12) so every probe in one report MUST share
# one order. The menu itself is a report axis: rrj's deep prior absorbs the
# enemy-presence flip in core/hop menus, so presence-trigger tests use mix.
MENUS: dict[str, tuple[Action, ...]] = {
    "core": (
        Action.RIGHT_JUMP,
        Action.RIGHT,
        Action.RIGHT_RUN,
        Action.RIGHT_RUN_JUMP,
        Action.LEFT,
    ),
    "mix": (Action.RIGHT_JUMP, Action.RIGHT_RUN, Action.LEFT),
    "step": (Action.RIGHT_JUMP, Action.LEFT),
    # the "natural fall" yield question: no tested gameplay menu ever offered
    # NOOP, so releasing the stick mid-air was inexpressible. This menu gives
    # the model exactly the choice the observation describes.
    "release": (Action.RIGHT_JUMP, Action.NOOP, Action.LEFT),
}
JUMP_FAMILY = {Action.RIGHT_JUMP, Action.RIGHT_RUN_JUMP, Action.JUMP}
ADVANCE_FAMILY = {Action.RIGHT, Action.RIGHT_RUN}

# enemy band edges in pixels (nearest ahead enemy)
NEAR, MID = 48, 112
# enemy contact frames below which jumping is the correct action
CONTACT_JUMP_MAX = 24
# contacts in this band are ambiguous (jump may cross low) -- excluded
CONTACT_AMBIGUOUS_MAX = 48
# per-category cap so one long bounce loop cannot dominate a bucket
PER_CATEGORY_CAP = 80


def _sig(s: MarioSnapshot) -> tuple:
    haz = s.threat_features()
    dist = haz.get("nearest_enemy_distance_pixels")
    nav = s.navigation_features()
    return (
        s.world, s.stage, s.x // 16, s.grounded,
        None if dist is None else dist // 16,
        nav.get("gap_distance_tiles"),
        nav.get("obstacle_distance_tiles"),
    )


def categorize(s: MarioSnapshot) -> tuple[str, dict]:
    """Return (bucket, meta) for a state; bucket '' == not usable."""
    haz = s.threat_features()
    nav = s.navigation_features()
    meta: dict = {"x": s.x, "world": s.world, "stage": s.stage}
    # airborne descent: the "yield" question -- mid-fall with an enemy on the
    # landing path, does the model stop holding right? (discrimination-only;
    # stomp-vs-yield both being legal, no accuracy label)
    if not s.grounded:
        enemy = bool(haz.get("enemy_ahead"))
        dist = haz.get("nearest_enemy_distance_pixels")
        if enemy and dist is not None and dist <= 64 and s.jump_phase == "falling":
            return "AIR_E", meta
        if not enemy and s.jump_phase == "falling":
            return "AIR_OPEN", meta
        return "", {}
    enemy = bool(haz.get("enemy_ahead"))
    dist = haz.get("nearest_enemy_distance_pixels")
    contact = haz.get("estimated_contact_frames")
    meta["enemy"], meta["dist"], meta["contact"] = enemy, dist, contact
    verdict = gap_takeoff_verdict(s)
    meta["verdict"] = verdict
    # wall = tall obstacle at conversational distance, no gap verdict
    wall = (
        verdict is None
        and nav.get("obstacle_distance_tiles") is not None
        and nav["obstacle_distance_tiles"] <= 2
        and (nav.get("obstacle_height_tiles") or 0) >= 3
    )
    meta["wall"] = wall
    if enemy:
        if dist is None:
            return "", meta
        if dist <= NEAR:
            return "E_NEAR", meta
        if dist <= MID:
            return "E_MID", meta
        return "E_FAR", meta
    if verdict == "now":
        return "G_NOW", meta
    if verdict == "too_early":
        return "G_EARLY", meta
    if verdict == "short":
        return "G_SHORT", meta
    if wall:
        return "W_TALL", meta
    return "OPEN", meta


def ground_truth(bucket: str, meta: dict) -> Action | None:
    """Parser-arithmetic correct family as a single representative action."""
    if bucket in ("AIR_E", "AIR_OPEN"):
        return None  # stomp and yield are both legal mid-air; no label
    if bucket == "G_NOW":
        return Action.RIGHT_JUMP
    if bucket in ("G_EARLY", "G_SHORT"):
        return Action.RIGHT
    if bucket in ("E_NEAR", "E_MID", "E_FAR"):
        contact = meta.get("contact")
        if contact is None or CONTACT_JUMP_MAX < contact <= CONTACT_AMBIGUOUS_MAX:
            return None  # ambiguous window, excluded from accuracy
        if contact is not None and contact <= CONTACT_JUMP_MAX:
            return Action.RIGHT_JUMP
        return Action.RIGHT
    if bucket == "W_TALL":
        return None  # no in-menu correct action (retreat-run-jump is a plan)
    return Action.RIGHT  # OPEN: advance is correct, jump is wasted


def harvest(paths: list[str]) -> list[MarioSnapshot]:
    seen: set[tuple] = set()
    out: list[MarioSnapshot] = []
    files: list[str] = []
    for p in paths:
        if Path(p).is_dir():
            files.extend(glob.glob(str(Path(p) / "**" / "*.jsonl"), recursive=True))
        else:
            files.extend(glob.glob(p))
    for f in sorted(files):
        try:
            lines = open(f).readlines()
        except OSError:
            continue
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("state") or not rec.get("debug_state"):
                continue
            try:
                snap = snapshot_from_record(rec)
            except Exception:  # noqa: BLE001 - old logs may drift in schema
                continue
            sig = _sig(snap)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(snap)
    return out


def run(paths: list[str], menu_name: str = "core") -> int:
    from .policy import LayaPolicy

    snaps = harvest(paths)
    buckets: dict[str, list[tuple[MarioSnapshot, dict]]] = {}
    for s in snaps:
        bucket, meta = categorize(s)
        if not bucket:
            continue
        buckets.setdefault(bucket, []).append((s, meta))
    for b in list(buckets):
        if len(buckets[b]) > PER_CATEGORY_CAP:
            step = len(buckets[b]) / PER_CATEGORY_CAP
            buckets[b] = [buckets[b][int(i * step)] for i in range(PER_CATEGORY_CAP)]

    menu = MENUS[menu_name]
    policy = LayaPolicy("models/hub/laya-multilingual-mlx", lang="zh", mode="pure")
    rows = []
    correct = Counter()
    total_labeled = Counter()
    choice_counts: dict[str, Counter] = {}
    try:
        for bucket, items in sorted(buckets.items()):
            jump_ps: list[float] = []
            picks: list[Action] = []
            for s, meta in items:
                d = policy.choose(s, menu)
                jump_ps.append(sum(
                    p for a, p in d.probabilities.items()
                    if a in {x.value for x in JUMP_FAMILY}
                ))
                picks.append(d.action)
                gt = ground_truth(bucket, meta)
                if gt is not None:
                    total_labeled[bucket] += 1
                    gt_family = JUMP_FAMILY if gt in JUMP_FAMILY else (
                        ADVANCE_FAMILY if gt in ADVANCE_FAMILY else {gt}
                    )
                    if d.action in gt_family:
                        correct[bucket] += 1
            cc = Counter(a.value for a in picks)
            choice_counts[bucket] = cc
            mu = mean(jump_ps)
            sd = pstdev(jump_ps) if len(jump_ps) > 1 else 0.0
            rows.append((bucket, len(items), mu, sd, cc))
    finally:
        policy.close()

    print(f"harvested {len(snaps)} unique states, menu={menu_name}={[a.value for a in menu]}")
    print(f"{'bucket':<8} {'n':>4} {'P(jump)':>8} {'±sd':>6}  top choices")
    for bucket, n, mu, sd, cc in rows:
        top = ", ".join(f"{k}:{v}" for k, v in cc.most_common(3))
        print(f"{bucket:<8} {n:>4} {mu:>8.3f} {sd:>6.3f}  {top}")

    labeled_all = sum(total_labeled.values())
    if labeled_all:
        acc_model = sum(correct.values()) / labeled_all
        # per-item baselines on the same labeled set
        import random as _r

        rng = _r.Random(0)
        always_jump = always_adv = random_acc = 0.0
        n = 0
        for bucket, items in buckets.items():
            for _, meta in items:
                gt = ground_truth(bucket, meta)
                if gt is None:
                    continue
                n += 1
                gt_family = JUMP_FAMILY if gt in JUMP_FAMILY else (
                    ADVANCE_FAMILY if gt in ADVANCE_FAMILY else {gt}
                )
                menu_jump = [a for a in menu if a in JUMP_FAMILY]
                menu_adv = [a for a in menu if a in ADVANCE_FAMILY]
                if menu_jump:
                    always_jump += rng.choice(menu_jump) in gt_family
                if menu_adv:
                    always_adv += rng.choice(menu_adv) in gt_family
                random_acc += rng.choice(list(menu)) in gt_family
        print(f"\naccuracy vs geometric ground truth ({n} labeled states):")
        print(f"  model          {acc_model:.3f}")
        print(f"  always-jump    {always_jump / n:.3f}")
        print(f"  always-advance {always_adv / n:.3f}")
        print(f"  random-menu    {random_acc / n:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="run jsonl files or directories")
    ap.add_argument("--menu", choices=sorted(MENUS), default="core",
                    help="pinned action menu (rrj-heavy menus mask the enemy-presence flip)")
    args = ap.parse_args(argv)
    return run(args.paths, menu_name=args.menu)


if __name__ == "__main__":
    sys.exit(main())
