# 旧代码清理与现行版本转正设计

> 状态: 已确认(2026-08-15 头脑风暴, 用户选定方案 A: 一次清理到位)。
> 背景: 旧管道(v1-v3)与现行 v4 事件管道长期并行, 用户决定删除旧代码、
> 现行版本转正为唯一版本。项目无 git, 用户明确选择不备份直接删。

## 1. 目标

代码树只保留现行事件流管道及其依赖; 旧运行时管道(旧入口、
GameSession、GameStateTracker、旧 HUD)全部删除; 现行代码去 v4 化转正名。
训练链(脚本 + 数据)全保留。每步全量测试兑底。

## 2. 删除清单

| 对象 | 说明 |
|---|---|
| `scripts/run_assistant.py` | 旧入口(446 行, 现行入口转正名后入住此名) |
| `src/mahjong_ui/format.py`、`src/mahjong_ui/pil_text.py` | 旧 HUD 文本渲染 |
| `src/mahjong_cv/yolo_mapping.py` | 旧检测映射(无引用) |
| `src/mahjong_ai/action_advisor.py` | 只被 session.py 旧循环使用(核实无其他引用) |
| `session.py` 内 GameSession 及专属依赖 | 无外部引用 |
| `state/tracker.py` 内 GameStateTracker(整文件) | 数据类先迁出(见 §3) |
| `tests/test_ui/`、`tests/test_state/` | 旧 UI/状态机测试 |
| `tests/test_ai/test_action_advisor.py` | 对应已删模块; test_ai 其余保留(现行 discard/效率测试) |
| `tests/test_placeholder.py` | 空占位 |
| config 中旧管道专用配置文件 | 实现时核实无引用则删(river_regions.json 等现行配置必须留) |

保留: pipeline 之外全部现行模块(inference/efficiency/strategy/client/
det_cluster/screen_vision/game_area/引擎规则含 shaanxi_rules 备用规则)、
全部训练/工具脚本与数据、docs(全部历史)、.superpowers 账本、
历史日志 data/settle_logs/game_v4_*.jsonl(数据不重命名)。

## 3. 拆分迁移

- **state/snapshot.py(新)**: 从 tracker.py 逐段搬移 `GameSnapshot`、
  `Meld`、`PlayerState`、`DiscardEvent`、`RIVER_NAMES` 及 GameSnapshot
  依赖的 `TURN_UNKNOWN`(核实定义位置后随迁), 内容一字不改。
  tracker.py 整文件删除; `state/__init__.py` re-export snapshot。
- **session.py 裁剪**: 删 GameSession, 留 `Advice`/`InferenceResult`/
  `OPPONENT_PLAYERS` — settle_check 与现行 runner 的 import 路径不变。
- import 同步改 7 处: opponent_inference、soft_observer、client、
  run_assistant、settle_check、pick_anchors、visualize_cluster(后两个是
  保留的训练工具, 只改 import 行不碰逻辑)。

## 4. 转正名

- `src/mahjong_ai/v4/` → `src/mahjong_ai/pipeline/`(8 模块); 全仓
  `src.mahjong_ai.v4.` → `src.mahjong_ai.pipeline.`(模块内部引用、
  runner、observer、tests 同步)
- `scripts/run_assistant_v4.py` → `scripts/run_assistant.py`(旧入口先删,
  名字让位); `scripts/dump_v4_log.py` → `scripts/dump_log.py`
- `tests/test_v4/` → `tests/test_pipeline/`
- 现行文件文档字样机械替换: "v4 管道" → "事件流管道"、"V4Runner" →
  "AssistantRunner" 等(不触碰 docs/ 历史文档与历史日志文件名)

## 5. 执行顺序(无 git, 顺序执行, 每步验绿再走)

1. 删纯旧文件(旧入口、format/pil_text、yolo_mapping、action_advisor、
   旧测试) → 全量测试
2. session.py 裁剪 → 全量测试
3. snapshot.py 迁移 + tracker.py 删 + 7 处 import 更新 → 全量测试
4. v4 → pipeline 目录改名 + import 全仓更新 → 全量测试
5. 入口/脚本/测试目录转正名 → 全量测试
6. 文档字样替换 + config 孤儿文件核实清理
7. 全量门禁 + 冒烟(客户端 import、settle_check/dump_log 运行、
   训练工具 import 不炸)+ 账本

## 6. 验证口径

- pytest 全量: 除 tests/test_cv 的 1 个预存失败(ROI 分类器测试,
  非本次引入)外全绿
- ruff/mypy: tracker.py 删除后 mypy 旧债自然消失; 口径保持"现行文件
  干净, 旧债不修"(清理后旧债应已随删除消失)
- 冒烟: 客户端 import、settle_check/dump_log 可运行、保留的训练脚本
  import 成功(不跑 GUI/训练)
