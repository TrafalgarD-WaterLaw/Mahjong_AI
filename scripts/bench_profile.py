"""性能基准报告 — 从对局日志提取帧耗时统计与事件密度。

输入: data/settle_logs/game_*.jsonl
      (run_assistant.py 每 5 帧写一条 frame_time 记录, 其余类型用于事件统计)
输出: 控制台表格 + docs/BENCHMARK.md + runs/bench/ 图表(matplotlib 可选)

用法:
    uv run python scripts/bench_profile.py
    uv run python scripts/bench_profile.py --log-dir data/settle_logs
    uv run python scripts/bench_profile.py --no-plots
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

#: Windows 控制台默认 GBK, 强制 UTF-8 输出避免中文乱码
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

#: frame_time 记录中的耗时字段(毫秒换算在统计时做)
_TIME_FIELDS = ('t_capture', 't_recognize', 't_handle')
#: 事件统计关注的记录类型
_EVENT_TYPES = (
    'TileAppeared', 'HandChanged', 'MeldFormed', 'TileClaimed',
    'TileVanished', 'FlowerShown', 'freeze', 'risk', 'infer',
)
#: 识别长尾阈值(秒) — 超过记为慢帧
_SLOW_RECOGNIZE = 0.5
#: 帧间隔(秒, QTimer 66ms → 约 15fps 目标)
_TICK_SEC = 0.066


@dataclass
class LogStats:
    """单局日志的统计结果。"""

    name: str
    duration: float = 0.0              # 日志跨度(秒)
    frames: list[float] = field(default_factory=list)   # frame 序号
    ts: list[float] = field(default_factory=list)       # 记录时间戳
    times: dict[str, list[float]] = field(default_factory=dict)
    queue_len: list[int] = field(default_factory=list)
    events: Counter[str] = field(default_factory=Counter)
    n_frame_time: int = 0
    n_lines: int = 0

    def __post_init__(self) -> None:
        for f in _TIME_FIELDS:
            self.times.setdefault(f, [])


def parse_log(path: Path) -> LogStats:
    """解析单个 jsonl: 收集 frame_time 与事件计数。"""
    st = LogStats(name=path.name)
    t0: float | None = None
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        st.n_lines += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        typ = rec.get('type', '')
        if typ == 'frame_time':
            st.n_frame_time += 1
            st.frames.append(float(rec.get('frame', 0)))
            ts = float(rec.get('ts', 0.0))
            st.ts.append(ts)
            for f in _TIME_FIELDS:
                st.times[f].append(float(rec.get(f, 0.0)))
            st.queue_len.append(int(rec.get('queue', 0)))
            if t0 is None:
                t0 = ts
        elif typ in _EVENT_TYPES:
            st.events[typ] += 1
    if t0 is not None and st.ts:
        st.duration = st.ts[-1] - t0
    return st


def percentile(sorted_vals: list[float], p: float) -> float:
    """p 分位(0-1), 输入需已排序。空列表返回 0。"""
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
    return sorted_vals[i]


def fmt_ms(v: float) -> str:
    return f'{v * 1000:.0f}ms' if v >= 0.1 else f'{v * 1000:.1f}ms'


def row(field: str, vals: list[float]) -> str:
    """统计表格的一行: 均值 / P50 / P95 / P99 / 最大。"""
    s = sorted(vals)
    mean = statistics.fmean(vals) if vals else 0.0
    return (f'  {field:<12} {fmt_ms(mean):>9} {fmt_ms(percentile(s, 0.5)):>8} '
            f'{fmt_ms(percentile(s, 0.95)):>8} {fmt_ms(percentile(s, 0.99)):>8} '
            f'{fmt_ms(s[-1] if s else 0.0):>8}')


def fps_of(st: LogStats) -> float | None:
    """帧率: (帧序号差) / (墙钟时间差)。记录稀疏不足两跳 → None。"""
    if len(st.frames) < 2 or st.duration <= 0:
        return None
    return (st.frames[-1] - st.frames[0]) / st.duration


def gpu_info() -> str:
    """nvidia-smi 可用时返回 GPU 型号; 否则 '未知'。"""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return '未知'


def machine_info() -> str:
    parts = [platform.system(), platform.machine()]
    proc = os.environ.get('PROCESSOR_IDENTIFIER')
    if proc:
        parts.append(proc)
    return ' / '.join(parts)


def render_console(all_stats: list[LogStats], merged: LogStats) -> str:
    """控制台摘要文本。"""
    lines: list[str] = []
    lines.append(f'=== 性能基准: {len(all_stats)} 局, '
                 f'{merged.n_frame_time} 条 frame_time 记录 ===')
    lines.append('阶段           均值      P50      P95      P99    最大')
    for f in _TIME_FIELDS:
        lines.append(row(f, merged.times[f]))
    q = merged.queue_len
    lines.append(f'  队列积压     queue>0 占 '
                 f'{100.0 * sum(1 for v in q if v > 0) / len(q):.1f}%'
                 if q else '  队列积压     (无样本)')
    rec = merged.times['t_recognize']
    slow = 100.0 * sum(1 for v in rec if v > _SLOW_RECOGNIZE) / len(rec) \
        if rec else 0.0
    lines.append(f'  识别慢帧     >{_SLOW_RECOGNIZE * 1000:.0f}ms 占 {slow:.1f}%')
    lines.append('')
    lines.append('=== 分局帧率 ===')
    for st in all_stats:
        fps = fps_of(st)
        fps_s = f'{fps:.1f} fps' if fps else '样本不足'
        lines.append(f'  {st.name:<42} {fps_s:>10}  '
                     f'{st.duration:.0f}s  {st.n_frame_time} 条')
    return '\n'.join(lines)


def render_markdown(all_stats: list[LogStats], merged: LogStats,
                    out_path: Path, log_dir: Path) -> str:
    """docs/BENCHMARK.md 报告。"""
    q = merged.queue_len
    rec = merged.times['t_recognize']
    slow_pct = 100.0 * sum(1 for v in rec if v > _SLOW_RECOGNIZE) / len(rec) \
        if rec else 0.0
    queue_pct = 100.0 * sum(1 for v in q if v > 0) / len(q) if q else 0.0

    # 自动观察: 主瓶颈 + 长尾 + 积压
    means = {f: statistics.fmean(merged.times[f]) if merged.times[f] else 0.0
             for f in _TIME_FIELDS}
    order = sorted(_TIME_FIELDS, key=lambda f: -means[f])
    bottleneck, second = order[0], order[1]
    rec_s = sorted(rec)
    tail_ratio = rec_s[-1] / rec_s[0] if len(rec_s) > 1 else 1.0
    observations = [
        f'- 耗时占比最大的是 **{bottleneck}**(均值 {fmt_ms(means[bottleneck])}),'
        f' 其次 {second}({fmt_ms(means[second])})。',
        f'- 识别耗时 P99/P50 = {tail_ratio:.1f}x'
        + (f', 存在明显长尾(P99 {fmt_ms(percentile(rec_s, 0.99))})'
           if tail_ratio > 5 else ', 分布较稳定'),
        f'- 观测队列积压率 {queue_pct:.1f}%'
        + (' → 识别偶尔跟不上主循环' if queue_pct > 0 else ''),
        f'- 识别慢帧(>{_SLOW_RECOGNIZE * 1000:.0f}ms)占比 {slow_pct:.1f}%',
    ]
    now = datetime.now(UTC).astimezone().strftime('%Y-%m-%d %H:%M')

    m: list[str] = []
    m.append('# 性能基准报告\n')
    m.append(f'- 生成时间: {now}')
    m.append(f'- 日志范围: {len(all_stats)} 局, {merged.n_frame_time} 条 frame_time 记录')
    m.append(f'- 环境: {machine_info()}')
    m.append(f'- GPU: {gpu_info()}\n')
    m.append('## 汇总(跨局合并)\n')
    m.append('| 阶段 | 均值 | P50 | P95 | P99 | 最大 |')
    m.append('|------|------|-----|-----|-----|------|')
    for f in _TIME_FIELDS:
        s = sorted(merged.times[f])
        mean = statistics.fmean(s) if s else 0.0
        m.append(f'| {f} | {fmt_ms(mean)} | {fmt_ms(percentile(s, 0.5))} '
                 f'| {fmt_ms(percentile(s, 0.95))} '
                 f'| {fmt_ms(percentile(s, 0.99))} '
                 f'| {fmt_ms(s[-1] if s else 0.0)} |')
    m.append(f'| 队列积压 | queue>0 占 {queue_pct:.1f}% | - | - | - | - |')
    m.append(f'| 识别慢帧 | >{_SLOW_RECOGNIZE * 1000:.0f}ms 占 {slow_pct:.1f}% | - | - | - | - |\n')
    m.append('## 分局明细\n')
    m.append('| 日志 | 帧率 | 跨度 | 样本 | 事件(出现/手牌/副露/冻结) |')
    m.append('|------|------|------|------|------------------------|')
    for st in all_stats:
        fps = fps_of(st)
        fps_s = f'{fps:.1f} fps' if fps else '-'
        ev = st.events
        m.append(f'| {st.name} | {fps_s} | {st.duration:.0f}s | '
                 f'{st.n_frame_time} | '
                 f'{ev.get("TileAppeared", 0)} / {ev.get("HandChanged", 0)} / '
                 f'{ev.get("MeldFormed", 0)} / {ev.get("freeze", 0)} |')
    m.append('\n## 事件密度(跨局)\n')
    if merged.events:
        m.append('| 类型 | 数量 | 每局均值 |')
        m.append('|------|------|----------|')
        for typ in _EVENT_TYPES:
            n = merged.events.get(typ, 0)
            if n:
                m.append(f'| {typ} | {n} | {n / len(all_stats):.1f} |')
    m.append('\n## 观察\n')
    m.extend(observations)
    m.append('')
    m.append(f'> 报告由 `uv run python scripts/bench_profile.py '
             f'--log-dir {log_dir}` 生成')
    return '\n'.join(m)


def plot_report(all_stats: list[LogStats], out_dir: Path) -> bool:
    """matplotlib 可用时生成图表; 否则返回 False。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        return False
    # 中文字体回退(Windows: 微软雅黑/黑体; 其他平台跳过)
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC'):
        if name in installed:
            matplotlib.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 三阶段耗时分布箱线图
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    if all_stats:
        data = [[v * 1000 for v in all_stats[0].times[f]] for f in _TIME_FIELDS]
        for st in all_stats[1:]:
            for i, f in enumerate(_TIME_FIELDS):
                data[i] += [v * 1000 for v in st.times[f]]
    ax.boxplot(data, tick_labels=_TIME_FIELDS, showfliers=False)
    ax.set_ylabel('耗时 (ms)')
    ax.set_title('各阶段耗时分布 (不含离群点)')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'latency_boxplot.png', dpi=150)
    plt.close(fig)

    # 2) 识别耗时随时间曲线(每局一条)
    fig, ax = plt.subplots(figsize=(10, 5))
    for st in all_stats:
        if not st.ts:
            continue
        t0 = st.ts[0]
        ax.plot([t - t0 for t in st.ts],
                [v * 1000 for v in st.times['t_recognize']],
                lw=0.8, label=st.name[-15:])
    ax.axhline(_SLOW_RECOGNIZE * 1000, color='r', ls='--', lw=1,
               label=f'慢帧阈值 {_SLOW_RECOGNIZE * 1000:.0f}ms')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('识别耗时 (ms)')
    ax.set_title('识别耗时随时间变化 (长尾/退化检测)')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'recognize_over_time.png', dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description='对局日志性能基准报告')
    parser.add_argument('--log-dir', default='data/settle_logs')
    parser.add_argument('--out', default='docs/BENCHMARK.md')
    parser.add_argument('--plot-dir', default='runs/bench')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f'日志目录不存在: {log_dir}')
        raise SystemExit(1)
    files = sorted(
        p for p in log_dir.glob('game_*.jsonl') if p.stat().st_size > 0)
    if not files:
        print(f'{log_dir} 下没有非空 game_*.jsonl 日志')
        raise SystemExit(1)

    all_stats = [parse_log(p) for p in files]
    all_stats = [st for st in all_stats if st.n_frame_time > 0]
    if not all_stats:
        print('所有日志都没有 frame_time 记录(需要 run_assistant.py 新版日志)')
        raise SystemExit(1)

    merged = LogStats(name='(all)')
    for st in all_stats:
        merged.duration += st.duration
        merged.n_frame_time += st.n_frame_time
        merged.n_lines += st.n_lines
        merged.events.update(st.events)
        for f in _TIME_FIELDS:
            merged.times[f].extend(st.times[f])
        merged.queue_len.extend(st.queue_len)

    console = render_console(all_stats, merged)
    print(console)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown(all_stats, merged, out_path, log_dir)
    out_path.write_text(md, encoding='utf-8')
    print(f'\n报告已写入: {out_path}')

    if not args.no_plots:
        ok = plot_report(all_stats, Path(args.plot_dir))
        if ok:
            print(f'图表已写入: {Path(args.plot_dir)}')
        else:
            print('matplotlib 未安装, 跳过图表'
                  '(安装: uv add --dev matplotlib)')


if __name__ == '__main__':
    main()
