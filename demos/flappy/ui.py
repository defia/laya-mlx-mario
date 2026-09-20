"""Rich terminal rendering: canvas on the left, live decision panel on the right."""

from __future__ import annotations

import math

from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .game import BIRD_X, FlappyGame
from .player import Decision

SKY = " "
PIPE_CHAR = "█"
GROUND_CHAR = "▂"
BIRD_CHAR = "◉"


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


def canvas(game: FlappyGame, note: str = "") -> Panel:
    body_lines: list[Text] = []
    for y in range(game.rows):
        if y == game.rows - 1:
            body_lines.append(Text(GROUND_CHAR * game.cols, style="dim yellow"))
            continue
        chars = [SKY] * game.cols
        spans: list[tuple[str, int, int]] = []
        for pipe in game.pipes:
            if pipe.x + 3 <= 0 or pipe.x >= game.cols:
                continue
            if y < pipe.gap_top or y >= pipe.gap_bottom:
                for x in range(max(0, pipe.x), min(game.cols, pipe.x + 3)):
                    chars[x] = PIPE_CHAR
                spans.append(("green", max(0, pipe.x), min(game.cols, pipe.x + 3)))
        if game.alive and y == game.bird_y:
            chars[BIRD_X] = BIRD_CHAR
            spans.append(("bold yellow", BIRD_X, BIRD_X + 1))
        text = Text("".join(chars))
        for style, start, end in spans:
            text.stylize(style, start, end)
        body_lines.append(text)
    body = Text("\n").join(body_lines)
    title = f"得分 {game.score}   步数 {game.steps}"
    if note:
        title += f"   ·  {note}"
    return Panel(body, title=title, border_style="cyan", padding=(0, 0))


def panel(game: FlappyGame, decision: Decision | None, stats: dict) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", justify="right", width=10)
    table.add_column(ratio=1)

    table.add_row("局数", f"{stats['episodes']}    最佳 {stats['best']}")
    vy = game.vy
    vy_text = f"+{-vy}↑" if vy < 0 else f"{vy}↓"
    pipe = game.next_pipe
    gap_text = f"第{pipe.gap_center}行 · 距{game.columns_to_gap()}列" if pipe else "—"
    table.add_row("鸟", f"第{game.bird_y}行  速度{vy_text}")
    table.add_row("缺口", gap_text)
    table.add_row("", Text("─" * 26, style="dim"))

    if decision is not None:
        p = decision.probabilities
        jump_p = float(p.get("jump", 0.0))
        glide_p = float(p.get("glide", 0.0))
        marker = " ✓盾" if decision.shielded else (" ←" if decision.action == "jump" else "")
        table.add_row("拍翅", bar(jump_p) + f" {jump_p:4.0%}{marker if decision.action == 'jump' else ''}")
        table.add_row(
            "滑翔",
            bar(glide_p) + f" {glide_p:4.0%}{' ←' if decision.action == 'glide' and not decision.shielded else ''}",
        )
        table.add_row("撞击风险", bar(decision.danger) + f" {decision.danger:4.0%}")
        table.add_row("紧急度", f"{decision.urgency:.1f} / 3")
        if decision.shielded:
            table.add_row("", Text("SHIELD 已纠正模型选择", style="bold magenta"))
        table.add_row("", Text("─" * 26, style="dim"))
        table.add_row("推理", f"{decision.inference_ms:5.1f} ms")
    table.add_row("决策率", f"{stats['rate']:.0f} /s   共 {stats['decisions']}")
    table.add_row("护盾干预", str(stats["shields"]))
    strategy = stats.get("strategy")
    if strategy:
        table.add_row("飞行策略", Text(strategy, style="bold yellow"))
    table.add_row("网络", Text("OFFLINE ✓ 本地推理", style="green"))
    keys = Text("Q 退出 · P 暂停")
    if strategy:
        keys = Text("1省力 2居中 3飞高 · Q 退出 · P 暂停")
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
        jp, gp = e["jump_p"], e["glide_p"]
        tail.append(f"拍{jp:3.0%}", style="bold yellow" if e["pick"] == "jump" else "dim yellow")
        tail.append(" ")
        tail.append(f"滑{gp:3.0%}", style="bold cyan" if e["pick"] == "glide" else "dim cyan")
        tail.append(" → ")
        verb = "拍翅" if e["exec"] == "jump" else "滑翔"
        tail.append(verb, style="bold green" if e["exec"] == e["pick"] else "bold magenta")
        if e["shielded"]:
            tail.append(" [SHIELD]", style="bold magenta")
        if e.get("died"):
            tail.append(f" [CRASH·{e['died']}]", style="bold red")
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
    game: FlappyGame,
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
    note = "" if game.alive else f"CRASH · {game.death_reason}"
    layout["board"].update(canvas(game, note))
    layout["side"].update(panel(game, decision, stats))
    layout["log"].update(DecisionLog(log or []))
    return layout


def live() -> Live:
    return Live(refresh_per_second=15, screen=True)
