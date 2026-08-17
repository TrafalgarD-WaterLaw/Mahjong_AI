# 麻将AI助手 (Mahjong AI Assistant)

面向真实游戏客户端的实时麻将辅助决策系统:通过截屏"看"牌桌,重建完整牌局状态,推断三家对手的隐藏手牌与听牌威胁,并给出牌效最优的出牌建议。

只读辅助 — 不操作游戏、不注入、不修改游戏进程。

## 核心能力

- **牌面识别**: YOLOv8 检测 + ByteTrack 身份跟踪 + ROI 分类器二级精修(牌河 95-99% precision / 74% recall, 分类 98.3%), 低阈值召回 + 背景类过滤
- **事件流归属解码**: 把"错误率 25% 的逐帧检测"抽象成可靠事件流(牌出现/消失/手牌台阶/副露信号), 轮转自动机 + beam search 全局归属解码"谁打了什么", 软归属 + 冻结窗口 + 翻案回放修正
- **对手隐藏手牌推断**: 粒子滤波(3 家 × 400 粒子)+ 牌效似然 + 结构化漂移, 输出听牌概率/最可能等待牌/放铳率; 副露内容用牌池约束反解(碰/吃/未知按碰近似)
- **出牌推荐**: 向听数优先 + 有效进张枚数(按剩余张数加权), 副露展开评估
- **自适应 UI**: 三档布局(窄条/牌桌/宽屏)+ 降级链, 任意窗口尺寸不挤不溢出

## 系统架构

```
游戏画面 ──截屏──▶ 检测 + 跟踪 + ROI 精修(逐框批量分类)
                        │  每帧 ~60 牌面框
                        ▼
          事件提取: 手牌带锚定 / 牌河消失检测 / 副露台阶签名
                        │  TileAppeared / TileVanished / HandChanged / MeldFormed
                        ▼
          轮转自动机 + beam search 全局归属解码(软归属 + 8s 冻结 + 翻案)
                        ▼
          粒子滤波对手建模: 听牌概率 / 等待牌 / 放铳率
                        ▼
          牌效引擎出牌推荐 ──▶ 三档自适应 UI 面板
```

模块划分:

| 目录 | 职责 |
|---|---|
| `src/mahjong_cv/` | 截屏捕获、YOLO 检测 + ByteTrack + ROI 分类、聚类 |
| `src/mahjong_ai/pipeline/` | 事件提取器、牌河消失检测、手牌账本、副露推断、归属解码器 |
| `src/mahjong_ai/inference/` | 粒子滤波对手建模(软观测、快照回放翻案、副露扣池) |
| `src/mahjong_ai/efficiency/` | 向听数计算、有效进张出牌推荐 |
| `src/mahjong_engine/` | 可扩展规则引擎(胡牌/听牌/动作判定) |
| `src/mahjong_ui/` | 独立信息面板(三档自适应布局) |
| `scripts/` | 运行入口、评估工具、数据标注/训练链 |
| `docs/superpowers/specs/` | 全部设计文档(架构决策与演化记录) |

## 推断准确性的离线验证

自建合成对局模拟器,真值对比评估(10 局 × 400 粒子):

- **手牌质量**: 对手真手牌每张的平均推断期望张数 0.95(随机基线 ≈ 0.11, 完美 = 1.0)
- **听牌校准**: 分桶预测概率 vs 实际听牌率,各桶实际值均落在预测区间内(单调)
- **等待牌命中**: 实际听牌时,真实等待牌在推断 top3 的比例 33-55%
- **出牌推荐 A/B**: 新算法(向听数 + 进张枚数)vs 旧算法,平均向听数全程更低

评估入口: `uv run python scripts/eval_inference.py`

## 快速开始

```bash
uv sync            # 安装依赖(含 GPU 版 torch)
uv run pytest      # 359 个测试
uv run mypy src    # 类型检查
uv run ruff check src tests scripts
```

运行实时辅助(需欢乐麻将 PC 客户端已开启):

```bash
uv run python scripts/run_assistant.py
```

> 模型权重与训练数据**不在本仓库**(`data/` 已排除)。目录结构约定:
> `data/models/screen/mahjong_screen_detector/weights/best.pt`(检测器)、
> `data/models/roi_cls/weights/best.pt`(ROI 分类器)。
> 训练链: `capture_dataset.py`(截图采集)→ `label_review.py`(人工审核)→ 训练。
> 首次运行按提示用 `pick_regions.py`(8 区域框选)完成桌面标定。

## 技术栈

| 层次 | 技术 |
|------|------|
| 语言/环境 | Python 3.12 / uv |
| 检测 | YOLOv8 + ByteTrack(ultralytics) |
| 牌面分类 | YOLOv8n-cls(ROI 二级精修) |
| 推断 | 粒子滤波(自实现)+ 轮转自动机 + beam search |
| UI | PySide6 (Qt) |
| 测试 | pytest / mypy / ruff |

## 已知限制(诚实边界)

- 检测 recall 冻结在 74%(数据闭环支持重训, 当前模型为基准)
- 对手等待牌推断受信息论限制: 藏牌从未出现时 top3 命中率 33-55%
- 目前只支持腾讯欢乐麻将 PC 客户端的桌面布局(区域标定方式可迁移)

## 免责声明

本项目仅用于计算机视觉与概率推断的工程学习与个人对局复盘。请勿用于任何违反游戏服务条款的行为;使用者自行承担相应责任。

## 许可

MIT — 见 [LICENSE](LICENSE)
