"""State text + typed questions, snake-style: planner verdicts baked into labels.

Same semantics as the other demos: the geometric planner predicts the receive
point (where the paddle should be when the ball arrives) and which moves still
catch the ball; those verdicts are stated in the option descriptions, and the
model genuinely chooses among the admissible actions.

A/B probing showed this checkpoint plays Breakout best with a factual,
strategy-free prompt — extra advisory sentences ("稳稳接住", "用拍面中央"…)
measurably hurt its directional accuracy. So strategies act purely through the
already-adjusted receive point stated in the facts, never through advice.
"""

from __future__ import annotations

from .game import BreakoutGame

STRATEGIES = {
    "steady": {"name": "稳接"},  # receive with the paddle center
    "steer": {"name": "控球"},  # receive with an edge, aiming at the denser side
}

INSTRUCTIONS = "球拍必须及时移到接球点正下方，选择最佳安全动作。"


def describe(game: BreakoutGame, ok: set[str], best: str, target_col: int | None, eta: int) -> str:
    d = "下" if game.vy > 0 else "上"
    s = (
        f"打砖块游戏。{len(ok)}/3个动作安全。"
        f"球在第{game.ball_y}行第{game.ball_x}列，向{'左' if game.vx < 0 else '右'}{d}飞。"
    )
    center = game.paddle_center
    if target_col is None:
        s += "接球点太远，暂不用移动。"
    else:
        dx = target_col - center
        if dx < -1:
            s += f"接球点在球拍中心左侧{-dx}列(第{target_col}列)，{eta}步后到达，应向左移动。"
        elif dx > 1:
            s += f"接球点在球拍中心右侧{dx}列(第{target_col}列)，{eta}步后到达，应向右移动。"
        else:
            s += f"接球点在球拍正上方(第{target_col}列)，{eta}步后到达，保持即可。"
    s += f"球拍中心在第{center}列。第{game.level}关剩{len(game.bricks)}块砖。"
    return s


def build_questions(ok: set[str], best: str) -> dict:
    best_labels = {
        "left": "安全。向左移动接球的最佳动作。",
        "right": "安全。向右移动接球的最佳动作。",
        "stay": "安全。已对准接球点，保持不动。",
    }
    # non-best options carry no direction words: they only say "not the best"
    off_labels = {
        "left": "安全。偏离接球点。",
        "right": "安全。偏离接球点。",
        "stay": "安全。偏离接球点。",
    }

    def label(action: str) -> str:
        if action not in ok:
            return "危险，球会落地漏接。" if ok else "没有安全动作，漏接已成定局。"
        return best_labels[action] if action == best else off_labels[action]

    return {
        "action": {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": {a: label(a) for a in ("left", "stay", "right")},
        },
        "danger": {
            "type": "noul",
            "instructions": "如果球拍保持不动，这一球会不会漏接落地？",
        },
        "urgency": {
            "type": "score",
            "instructions": "这一球的接球紧迫程度如何？",
            "criteria": ["从容", "正常", "紧迫", "火烧眉毛"],
        },
    }
