"""State text + typed questions, snake-style: planner verdicts baked into labels.

Mirrors the official Snake demo semantics: the geometric planner computes which
actions are safe and which one makes the best adjustment toward the gap; those
verdicts are stated in the option descriptions, and the model genuinely chooses
among them (one predict call also answers danger + urgency for the side panel).
"""

from __future__ import annotations

from .game import FlappyGame

STRATEGIES = {
    "lazy": {
        "name": "省力",
        "hint": "策略:节省体力，能滑翔就滑翔，及时拍翅避免危险。",
        # roll-out admissibility only: the bird may drift low between flaps
    },
    "center": {
        "name": "居中",
        "hint": "策略:始终从缺口中央通过，一旦低于中心立即拍翅回中。",
        "glide_ceiling": "center",  # glide allowed only while at/above gap center
    },
    "high": {
        "name": "飞高",
        "hint": "策略:保持高度，尽量贴着缺口上沿飞过，只允许小幅下滑。",
        "glide_ceiling": "top",  # glide allowed only while near the gap's top edge
    },
}


def describe(game: FlappyGame, ok: set[str], best: str, strategy: str = "lazy") -> str:
    bird_h = game.rows - 1 - game.bird_y
    s = f"Flappy游戏。{len(ok)}/2个动作安全。鸟离地{bird_h}行。"
    pipe = game.next_pipe
    if pipe is not None:
        gap_h = game.rows - 1 - pipe.gap_center
        diff = gap_h - bird_h
        rel = f"比鸟{'高' if diff > 0 else '低'}{abs(diff)}行" if diff else "与鸟同高"
        s += f"缺口中心离地{gap_h}行({rel})，还有{game.columns_to_gap()}步。"
    s += STRATEGIES[strategy]["hint"]
    return s


def build_questions(ok: set[str], best: str) -> dict:
    def label(action: str) -> str:
        if action not in ok:
            return "不安全，即将碰撞。" if ok else "没有安全动作，撞毁不可避免。"
        return "安全。朝缺口调整的最佳动作。" if action == best else "安全。偏离缺口。"

    return {
        "action": {
            "type": "choice",
            "instructions": "选择朝缺口调整最好的安全动作，避免碰撞。",
            "criteria": {"jump": label("jump"), "glide": label("glide")},
        },
        "danger": {
            "type": "noul",
            "instructions": "如果这一步不拍翅，鸟是否会撞上管道、地面或飞出顶端？",
        },
        "urgency": {
            "type": "score",
            "instructions": "当前局面的紧急程度如何？",
            "criteria": ["安全", "需留意", "危险", "千钧一发"],
        },
    }
