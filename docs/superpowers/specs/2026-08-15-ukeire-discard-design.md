# 出牌推荐算法重设计:向听数优先 + 进张枚数牌效

> 状态: 已确认(2026-08-15 头脑风暴三节全过)。旧代码不删, 新算法并行写入。

## 1. 目标

替换现行出牌推荐的排序核心。现行算法(向听数 + lonely 启发式)同向听
候选全靠单张"搭子潜力"近似, 不看进张面与剩余枚数 — 推荐不符合牌理。
新算法: **纯牌效**, 向听数优先 + **有效进张枚数** 第二优先(现代牌理
教科书算法), 放铳率不参与(用户确认)。

## 2. 现行问题清单(重设计的动机)

1. 同向听候选只靠 lonely 启发式整数, 不看全局结构
2. 没有进张枚数 — 真牌效 = 进张后向听改善 × 剩余枚数
3. 对子/两对子结构价值缺失
4. 七对/幺九型混排, 无型一致性
5. 备选分数公式(10 − 3×shanten + min(acceptance,5))是凑的
6. 无防守维度(按需求不加入 — 纯牌效)

## 3. 评分模型

对每个候选 t(唯一牌):

1. `shanten_after` = calculate_shanten(打出后手牌 + 副露展开)(第一优先,
   沿用现有计算, lru_cache 缓存)
2. `ukeire` = **有效进张总枚数**: 枚举所有 available[d] > 0 的牌 d,
   shanten(打出后 + d) < shanten_after → 计 available[d] 枚。
   进张面 12 的边张搭 vs 进张面 4 的孤张 — 算法自然拆搭留中张。
3. 并列时: `keep` = 保持向听的进张枚数(shanten 不变, 好形率近似)
4. 末位 tie-break: 现 lonely 启发式(保留, 兜底排序稳定)

排序: shanten_after ↑ → ukeire ↓ → keep ↓ → lonely ↑。

听牌(shanten_after == 0): acceptance = 等待牌总枚数(现逻辑沿用),
reason 显示"听 N 种 M 枚"。

## 4. 接口与数据流

- 新模块 `src/mahjong_ai/efficiency/ukeire_selector.py`(旧代码不删):
  - `recommend_by_ukeire(tiles, available, enabled_tiles, melds, exclude) -> DiscardRecommendation`
  - 复用 `DiscardRecommendation` dataclass; alternatives = 前 4 候选 +
    各自 ukeire 枚数
- `scripts/run_assistant.py` 的 build_advice 切换到新函数; `available`
  复用该函数内已计算的计数器(4 − 我手牌 − 各河 − 已亮副露)
- 客户端零改动
- 错误处理沿用: 2-14 张(含展开)约束 ValueError、全副露无候选 ValueError、
  副露展开/排除与现一致
- 性能: 候选 ≤14 × 进张 ≤34 × shanten(lru_cache)≈ 10-20ms/手

## 5. 理由文本

- 听牌: `打出 6条 后听牌, 听 3 种 11 枚`
- 未听: `打出 9饼 后一向听, 有效进张 28 枚`
- 副露/漏检标注沿用现行

## 6. 测试与验收

单元测试(经典牌效题, 断言具体出牌):
- 边张搭 1-2 vs 孤张 5 → 拆边张
- 字牌孤张最优先; 字牌对子 > 数牌孤张
- 两对子形态: 拆搭保两对
- 剩余枚数敏感: 5 万四张全可见 → 改选
- 听牌形态: acceptance = 等待牌总枚数
- 副露展开与排除回归

定量 A/B: eval_inference 的 SimPlayer 分别用新旧算法各 20 局,
比平均向听下降曲线 — 新算法应更快更低(定量证明"更符合牌理")。

实战验收: 用户真实对局观察推荐。

## 7. 范围边界

- 只做纯牌效(防守/危险度不进入推荐 — 用户确认)
- 1 层前瞻(用户确认); 不做 2 层搜索
- 旧 recommend_discard 保留不动(训练链/eval 兼容)
