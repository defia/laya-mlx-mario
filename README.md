# laya-mlx-mario

**本地类型化决策模型玩《超级马里奥兄弟》——零网络、零截图输入，模型只读状态事实文本，自己判断每个手拍该按什么键。**

本仓库 fork 自 [fhshaik/typesafe-mario](https://github.com/fhshaik/typesafe-mario)（把远程 TypeSafe/Jev API 换成本地 [laya-mlx](https://github.com/mizorewww/laya-mlx) MLX 运行时）。状态解析器、七个手柄宏动作、实时仪表盘、逐决策日志等上游骨架以 MIT 授权保留；被替换的只有"大脑"：

```text
上游：  NES 模拟器 → RAM/遥测解析器 → 结构化 JSON → TypeSafe Jev（云 API）→ 手柄输入
本仓库：NES 模拟器 → RAM/遥测解析器 → 事实性中文提示词 → laya-mlx（本地 MLX）→ 手柄输入
```

`Agent.predict` 一次前向同时回答三个类型化判定（choice 选动作 / noul 起跳判断 /
score 危险度），中文权重 ~15ms/决策，全程本地。

## 研究目标：pure 模式

不使用护盾、不告诉模型"该不该做什么"、没有任何警示标签——只给它运行基础信息
（自身运动、敌人位置/相对速度、地形读数、局部网格）和一段常驻游戏规则背景，
让模型自己判断，能玩得越远越好。

**当前结果：World 1-1 已通关**（中文权重 + 紧凑规则背景 + step 动作菜单 +
9帧/拍，240 个决策触旗，全程仅 108 帧卡墙）。

| 关卡 | 结果 | 卡点 |
|---|---|---|
| 1-1 | **通关**（240 拍） | 两面 4 格高墙的像素级抽签（已用措辞相位优化到 108 帧） |
| 1-2 | x=978 | 7 格高墙：原地跳极限 4 格，任何固定步态都无法通过的结构死点 |
| 1-3 | x=743 | 全程悬空台地，每跳需按间隙选跳跃力度——恰是模型"不读数字"的短板 |
| 1-4 | x=546 | 岩浆沟 + 火球棒（时机型障碍） |

## 快速开始

需要 Apple Silicon（MLX）+ Python 3.13。

```bash
uv sync --extra mario --extra demo

# 默认 = pure 模式 + 中文权重 + step 菜单 + 9帧/拍（1-1 通关配置）
uv run python -m demos.mario.cli play

# 无界面复现（轨迹完全确定，同配置逐步一致）
uv run python -m demos.mario.cli play --display none
```

仪表盘：`R` 重开 · `1`-`4` 随时切换 World 1-1~1-4 · `Esc`/`Q` 退出。
ROM 由 gym-super-mario-bros 包自带，仅限本地个人使用；本仓库不含任何任天堂数据。

## 主要发现（节选）

完整 21 条研究记录见 [demos/mario/README.md](demos/mario/README.md)。

- **模型 = 单步先验 + 强词汇锚定 + 不读数字**。敌距从 32px 压到 4px、接触倒计时
  从 16 帧压到 2 帧，选择分布纹丝不动——"告诉它来不及了"在信息层面就不成立。
- **pure 模式的突破全部来自动作空间设计**：从 7 键裁到 2 键（right_jump + left）
  的"常速短弧"步态，是通关 1-1 的步态；菜单里同时放 3 个跳跃选项时跳跃意图被
  摊薄，永远赢不了 argmax。
- **措辞即相位**：中文权重对每个 token 刀刃敏感——背景块任何诚实的改写
  （"110像素" vs "约7格"、语序、紧简）都会重掷整局轨迹。距离纪录是措辞搜索
  找出的、完全确定性的一条活相位；卡墙时长服从几何分布（同族措辞在同一面墙
  抽过 279 帧、2250 帧、甚至 5589 帧）。
- **地图的悖论**：ASCII 局部网格作为信息是死重（去掉后轨迹逐拍不变），但作为
  token 序列在承重（纪录相位依赖它的存在）。
- **英文权重完全不同**：分布尖峰、对任何文本变化免疫（措辞重掷无效），硬上限
  x=2466；中文权重分布平坦、刀刃敏感，反而靠"运气"通关。

## 仓库结构

mario 项目在 `demos/mario/`（解析器 `state.py`、提示词层 `describe.py`、
决策层 `policy.py`、运行器 `runner.py`、仪表盘 `dashboard.py`、CLI `cli.py`），
依赖的本地推理运行时在 `laya_mlx/`。仓库同时包含同一运行时的 snake/flappy/
breakout 终端 demo（`demos/`）。

## 致谢与授权

- [fhshaik/typesafe-mario](https://github.com/fhshaik/typesafe-mario)：上游原始
  逻辑（状态解析、动作映射、仪表盘、runner）以 MIT 授权引用。
- [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx)：本地 MLX 推理
  运行时（Apache-2.0）。
- [Convai Innovations laya](https://huggingface.co/convaiinnovations/laya) 系列
  预训练权重。
- 本仓库不含任何任天堂 ROM 或版权游戏数据，请自行确保模拟器与游戏文件的合法获取。
