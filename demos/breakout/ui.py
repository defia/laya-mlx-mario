"""Rich terminal rendering: board on the left, live decision panel on the right."""

from __future__ import annotations

import math

from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .game import BRICK_ROWS, BRICK_TOP, BRICK_W, PADDLE_ROW, PADDLE_W, BreakoutGame
from .player import Decision

BRICK_STYLES = ("bold red", "yellow", "green", "cyan", "magenta", "blue")
BRICK_CHAR = "█"
BALL_CHAR = "●"
PADDLE_CHAR = "▬"
DROP_CHAR = "▁"


def bar(prob: float, width: int = 12) -> Text:
    filled = max(0, min(width, round(prob * width)))
    text = Text("█" * filled + "░" * (width - filled))
    if prob >= 0.55:
        text.stylize("bold green")
    elif prob >= 0.3:
        text.stylize("yellow")
    else:
        text.stylize("dim")
    return text


def canvas(game: BreakoutGame, note: str = "") -> Panel:
    body_lines: list[Text] = []
    for y in range(game.rows):
        if y == game.rows - 1:
            body_lines.append(Text(DROP_CHAR * game.cols, style="dim red"))
            continue
        chars = [" "] * game.cols
        spans: list[tuple[str, int, int]] = []
        if BRICK_TOP <= y < BRICK_TOP + BRICK_ROWS:
            style = BRICK_STYLES[(y - BRICK_TOP) % len(BRICK_STYLES)]
            x = 0
            while x < game.cols:
                if (y, x // BRICK_W) in game.bricks:
                    start = x
                    while x < game.cols and (y, x // BRICK_W) in game.bricks:
                        chars[x] = BRICK_CHAR
                        x += 1
                    spans.append((style, start, x))
                else:
                    x += 1
        if y == PADDLE_ROW:
            for i in range(PADDLE_W):
                chars[game.px + i] = PADDLE_CHAR
            spans.append(("bold cyan", game.px, game.px + PADDLE_W))
        if game.alive and y == game.ball_y:
            chars[game.ball_x] = BALL_CHAR
            spans.append(("bold white", game.ball_x, game.ball_x + 1))
        text = Text("".join(chars))
        for style, start, end in spans:
            text.stylize(style, start, end)
        body_lines.append(text)
    body = Text("\n").join(body_lines)
    title = (
        f"得分 {game.score}   连击 {game.combo}   砖剩 {len(game.bricks)}   "
        f"命 {'♥' * game.lives}"
    )
    if note:
        title += f"   ·  {note}"
    return Panel(body, title=title, border_style="cyan", padding=(0, 0))


def panel(game: BreakoutGame, decision: Decision | None, stats: dict) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", justify="right", width=10)
    table.add_column(ratio=1)

    table.add_row("局数", f"{stats['episodes']}    最佳 {stats['best']}")
    table.add_row("关卡", f"第{game.level}关 · {game.steps}步")
    combo_style = "bold red" if game.combo >= 2 else "dim"
    table.add_row("连击", Text(f"当前 {game.combo} · 最高 {game.best_combo}", style=combo_style))
    arrow = ("←" if game.vx < 0 else "→") + ("↓" if game.vy > 0 else "↑")
    table.add_row("球", f"({game.ball_y},{game.ball_x})  {arrow}")
    if decision is not None and decision.landing is not None:
        table.add_row("落点", f"第{decision.landing}列 · 还有{decision.eta}步")
    else:
        table.add_row("落点", "—")
    table.add_row("", Text("─" * 26, style="dim"))

    if decision is not None:
        p = decision.probabilities
        lp = float(p.get("left", 0.0))
        sp = float(p.get("stay", 0.0))
        rp = float(p.get("right", 0.0))
        shield_mark = " ✓盾" if decision.shielded else ""
        table.add_row("左移", bar(lp) + f" {lp:4.0%}{' ←' if decision.action == 'left' else ''}")
        table.add_row("不动", bar(sp) + f" {sp:4.0%}{' ←' if decision.action == 'stay' else ''}")
        table.add_row(
            "右移", bar(rp) + f" {rp:4.0%}{' ←' if decision.action == 'right' else ''}{shield_mark if decision.action == 'right' else ''}"
        )
        if decision.shielded and decision.action != "right":
            table.add_row("", Text("SHIELD 已纠正模型选择", style="bold magenta"))
        table.add_row("漏接风险", bar(decision.danger) + f" {decision.danger:4.0%}")
        table.add_row("紧迫度", f"{decision.urgency:.1f} / 3")
        if decision.gains is not None:
            # the steer optimizer's per-step evaluation, shown live
            chosen = "center"
            if decision.landing is not None:
                if decision.gains["left"] > decision.gains["center"]:
                    chosen = "left"
                elif decision.gains["right"] > decision.gains["center"]:
                    chosen = "right"
            g = decision.gains
            text = Text()
            for key, name in (("left", "左沿"), ("center", "中央"), ("right", "右沿")):
                if key != "left":
                    text.append("  ")
                style = "bold green" if key == chosen and g[key] >= 0 else "dim"
                mark = " ✓" if key == chosen else ""
                text.append(f"{name}+{g[key]}{mark}", style=style)
            table.add_row("连击评估", text)
        table.add_row("", Text("─" * 26, style="dim"))
        table.add_row("推理", f"{decision.inference_ms:5.1f} ms")
    table.add_row("决策率", f"{stats['rate']:.0f} /s   共 {stats['decisions']}")
    table.add_row("护盾干预", str(stats["shields"]))
    strategy = stats.get("strategy")
    if strategy:
        table.add_row("接球策略", Text(strategy, style="bold yellow"))
    table.add_row("网络", Text("OFFLINE ✓ 本地推理", style="green"))
    keys = Text("Q 退出 · P 暂停")
    if strategy:
        keys = Text("1稳接 2控球 · Q 退出 · P 暂停")
    table.add_row("按键", keys)
    return Panel(table, title="Laya 决策", border_style="cyan")


def _entry_lines(entry: dict, width: int) -> int:
    """Rows one log entry needs: prompt (may wrap) + one metrics line."""
    head = 7  # "# 123 " prefix
    prompt_cells = head + cell_len(entry["prompt"])
    return max(1, math.ceil(prompt_cells / max(10, width))) + 1


def _build_log_panel(entries: list[dict], inner_w: int, inner_h: int) -> Panel:
    shown: list[dict] = []
    used = 0
    for e in reversed(entries):  # newest first, keep as many as fit
        need = _entry_lines(e, inner_w)
        if shown and used + need > inner_h:
            break
        shown.append(e)
        used += need
        if used >= inner_h:
            break
    shown.reverse()

    blocks: list[Text] = []
    for e in shown:
        block = Text()
        block.append(f"#{e['step']:>4} ", style="dim")
        block.append(e["prompt"], style="white")

        tail = Text()
        tail.append("      ")  # indent under the step number
        lp, sp, rp = e["left_p"], e["stay_p"], e["right_p"]
        tail.append(f"左{lp:3.0%}", style="bold yellow" if e["pick"] == "left" else "dim yellow")
        tail.append(" ")
        tail.append(f"停{sp:3.0%}", style="bold white" if e["pick"] == "stay" else "dim")
        tail.append(" ")
        tail.append(f"右{rp:3.0%}", style="bold cyan" if e["pick"] == "right" else "dim cyan")
        tail.append(" → ")
        verb = {"left": "左移", "stay": "不动", "right": "右移"}[e["exec"]]
        tail.append(verb, style="bold green" if e["exec"] == e["pick"] else "bold magenta")
        if e["shielded"]:
            tail.append(" [SHIELD]", style="bold magenta")
        if e.get("missed") is not None:
            tail.append(f" [漏接·剩{e['missed']}命]", style="bold red")
        if e.get("died"):
            tail.append(f" [GAMEOVER·{e['died']}]", style="bold red")
        if e.get("combo", 0) >= 2:
            tail.append(f" [连击{e['combo']}]", style="bold red")
        tail.append(f"  险{e['danger']:.0%} 急{e['urgency']:.1f} {e['ms']:.1f}ms", style="dim")

        block.append(Text("\n"))
        block.append(tail)
        blocks.append(block)
    body = Text()
    for i, b in enumerate(blocks):
        if i:
            body.append(Text("\n"))
        body.append(b)
    hidden = len(entries) - len(shown)
    title = "模型决策日志 · 上行=喂给模型的原文 下行=概率→执行"
    if hidden > 0:
        title += f"（更早 {hidden} 条已滚出）"
    return Panel(body, title=title, border_style="green", padding=(0, 1))


class DecisionLog:
    """Renderable log panel: re-measures its allotted box on every frame, so
    terminal resizes immediately change how many entries are shown."""

    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = options.max_width or console.size.width
        height = options.height or console.size.height
        inner_w = max(20, width - 4)  # panel borders + horizontal padding
        inner_h = max(2, height - 2)  # panel borders
        yield _build_log_panel(self.entries, inner_w, inner_h)


def render(
    game: BreakoutGame,
    decision: Decision | None,
    stats: dict,
    log: list[dict] | None = None,
) -> Layout:
    layout = Layout()
    board_h = game.rows + 2  # grid + panel borders
    board_w = game.cols + 2
    layout.split_column(
        Layout(name="top", size=board_h),
        Layout(name="log"),  # everything below the board goes to the log
    )
    layout["top"].split_row(
        Layout(name="board", size=board_w),
        Layout(name="side"),  # decision panel stretches with terminal width
    )
    note = "" if game.alive else f"GAME OVER · {game.death_reason}"
    layout["board"].update(canvas(game, note))
    layout["side"].update(panel(game, decision, stats))
    layout["log"].update(DecisionLog(log or []))
    return layout


def live() -> Live:
    return Live(refresh_per_second=15, screen=True)
