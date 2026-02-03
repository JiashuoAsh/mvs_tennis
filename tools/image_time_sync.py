"""三路相机图片时间同步与延迟分析工具。

该脚本面向如下目录结构（默认在工作目录下）：
- MV-CS050-02(DA8199285)
- MV-CS050-03(DA8199303)
- MV-CS050-CTRL(DA8199402)

核心思路（算法概述）：
1) 从文件名中解析时间戳（优先匹配 Image_YYYYMMDDHHMMSSxxx...）。
2) 将每路图片按时间排序，统计帧间隔 dt 分布，用于评估“卡顿/延迟/丢帧”。
3) 选择参考相机（默认 CTRL）。对其它相机做“单调最近邻匹配”（1-1 匹配）。
4) 计算匹配对的时间差 delta(t) = t_cam - t_ref：
   - offset：使用 delta 的中位数估计固定时间偏移（毫秒级对齐）。
   - jitter：残差 residual = delta - offset 的分布（抖动）。
   - drift：对 delta(t) 做一次线性回归，估计随时间漂移的斜率。
5) 输出 JSON 汇总 + CSV 细表，便于进一步绘图或排查。

注意：
- 本工具只依赖标准库，不读取图像内容；时间戳以文件名为准。
- 若文件名无法解析时间戳，会回退到文件修改时间（可能不可靠）。

运行示例：
    python tools/image_time_sync.py --root . --out tools_output

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean, median
from typing import Iterable, List, Optional, Sequence, Tuple


_IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageStamp:
    """图片与其时间戳。"""

    path: str
    ts: datetime


@dataclass(frozen=True)
class MatchPair:
    """两路相机的匹配对。"""

    ref: ImageStamp
    cam: ImageStamp
    delta_us: int


def _parse_compact_datetime(digits: str) -> Optional[datetime]:
    """解析紧凑型时间戳。

    支持：
    - YYYYMMDDHHMMSS
    - YYYYMMDDHHMMSSmmm (毫秒)
    - YYYYMMDDHHMMSSffffff (微秒)

    Args:
        digits: 仅包含数字的时间戳字符串。

    Returns:
        解析后的 datetime（naive）。无法解析返回 None。
    """

    if len(digits) < 14:
        return None

    base = digits[:14]
    frac = digits[14:]

    try:
        dt = datetime.strptime(base, "%Y%m%d%H%M%S")
    except ValueError:
        return None

    if not frac:
        return dt

    # 将小数部分转为微秒，长度不足补 0，过长截断。
    if len(frac) > 6:
        frac = frac[:6]
    else:
        frac = frac.ljust(6, "0")

    try:
        us = int(frac)
    except ValueError:
        return dt

    return dt + timedelta(microseconds=us)


def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """从文件名提取时间戳。

    优先匹配：Image_<digits>。

    Args:
        filename: 文件名（不含目录也可）。

    Returns:
        datetime 或 None。
    """

    m = re.search(r"Image_(\d{14,20})", filename)
    if m:
        return _parse_compact_datetime(m.group(1))

    # 容错：如果不是标准前缀，也尝试找最长的数字串。
    candidates = re.findall(r"\d{14,20}", filename)
    if not candidates:
        return None

    candidates.sort(key=len, reverse=True)
    for c in candidates:
        dt = _parse_compact_datetime(c)
        if dt is not None:
            return dt

    return None


def iter_image_files(folder: str, recursive: bool) -> Iterable[str]:
    """枚举目录内图片文件路径。"""

    if recursive:
        for root, _, files in os.walk(folder):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in _IMAGE_EXTS:
                    yield os.path.join(root, name)
        return

    with os.scandir(folder) as it:
        for entry in it:
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in _IMAGE_EXTS:
                yield entry.path


def load_timeline(folder: str, recursive: bool) -> List[ImageStamp]:
    """读取某一路相机目录，生成按时间排序的时间线。"""

    items: List[ImageStamp] = []
    for path in iter_image_files(folder, recursive=recursive):
        name = os.path.basename(path)
        ts = extract_timestamp_from_filename(name)
        if ts is None:
            # 回退到文件修改时间（可靠性取决于拷贝/解压流程）。
            ts = datetime.fromtimestamp(os.path.getmtime(path))
        items.append(ImageStamp(path=path, ts=ts))

    items.sort(key=lambda x: x.ts)
    return items


def _to_us(ts: datetime, base: datetime) -> int:
    return int((ts - base).total_seconds() * 1_000_000)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """线性插值分位数。q in [0, 100]。"""

    if not sorted_values:
        return float("nan")

    if q <= 0:
        return float(sorted_values[0])
    if q >= 100:
        return float(sorted_values[-1])

    k = (len(sorted_values) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])

    d0 = float(sorted_values[f]) * (c - k)
    d1 = float(sorted_values[c]) * (k - f)
    return d0 + d1


def compute_intervals_ms(timeline: Sequence[ImageStamp]) -> List[float]:
    """计算相邻帧的间隔（毫秒）。"""

    if len(timeline) < 2:
        return []

    out: List[float] = []
    for prev, cur in zip(timeline, timeline[1:]):
        out.append((cur.ts - prev.ts).total_seconds() * 1000.0)
    return out


def match_monotonic_nearest(
    ref: Sequence[ImageStamp],
    cam: Sequence[ImageStamp],
    max_abs_diff_ms: float,
    base: datetime,
) -> List[MatchPair]:
    """做单调(1-1)最近邻匹配。

    做法：
    - 将时间转换为相对 base 的整数微秒。
    - 对 cam 维护一个单调递增的起始下标 start_idx。
    - 对 ref 每个时间点，使用二分查找找到 cam 中最接近的点（不回退），
      若绝对误差 <= max_abs_diff_ms 则认为匹配成功。

    这能有效避免“一张图匹配到多张图”，也能在存在缺帧时保持稳定。
    """

    if not ref or not cam:
        return []

    ref_us = [_to_us(x.ts, base) for x in ref]
    cam_us = [_to_us(x.ts, base) for x in cam]

    import bisect

    max_abs_diff_us = int(max_abs_diff_ms * 1000.0)

    pairs: List[MatchPair] = []
    start_idx = 0

    for i, r_us in enumerate(ref_us):
        if start_idx >= len(cam_us):
            break

        pos = bisect.bisect_left(cam_us, r_us, lo=start_idx)

        candidates: List[int] = []
        if pos < len(cam_us):
            candidates.append(pos)
        if pos - 1 >= start_idx:
            candidates.append(pos - 1)

        if not candidates:
            continue

        best_idx = min(candidates, key=lambda j: abs(cam_us[j] - r_us))
        diff_us = cam_us[best_idx] - r_us

        if abs(diff_us) <= max_abs_diff_us:
            pairs.append(MatchPair(ref=ref[i], cam=cam[best_idx], delta_us=diff_us))
            start_idx = best_idx + 1

    return pairs


def _linear_regression_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """一元线性回归斜率（最小二乘）。"""

    if len(x) != len(y) or len(x) < 2:
        return float("nan")

    x_bar = mean(x)
    y_bar = mean(y)

    num = 0.0
    den = 0.0
    for xi, yi in zip(x, y):
        dx = xi - x_bar
        num += dx * (yi - y_bar)
        den += dx * dx

    if den == 0.0:
        return float("nan")

    return num / den


def summarize_deltas_ms(deltas_ms: Sequence[float]) -> dict:
    """汇总 delta（毫秒）统计。"""

    if not deltas_ms:
        return {
            "count": 0,
        }

    s = sorted(deltas_ms)
    return {
        "count": len(s),
        "mean_ms": mean(s),
        "median_ms": median(s),
        "p05_ms": _percentile(s, 5),
        "p95_ms": _percentile(s, 95),
        "p99_ms": _percentile(s, 99),
        "min_ms": s[0],
        "max_ms": s[-1],
    }


def summarize_abs_ms(values_ms: Sequence[float]) -> dict:
    """汇总 |x|（毫秒）统计，更适合描述抖动强度。"""

    return summarize_deltas_ms([abs(v) for v in values_ms])


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_intervals_csv(out_path: str, intervals_ms: Sequence[float]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "dt_ms"])
        for i, v in enumerate(intervals_ms):
            w.writerow([i, f"{v:.3f}"])


def write_matches_csv(out_path: str, pairs: Sequence[MatchPair], offset_ms: float) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ref_path",
                "ref_ts",
                "cam_path",
                "cam_ts",
                "delta_ms",
                "residual_ms",
            ]
        )
        for p in pairs:
            delta_ms = p.delta_us / 1000.0
            residual_ms = delta_ms - offset_ms
            w.writerow(
                [
                    p.ref.path,
                    p.ref.ts.isoformat(timespec="microseconds"),
                    p.cam.path,
                    p.cam.ts.isoformat(timespec="microseconds"),
                    f"{delta_ms:.3f}",
                    f"{residual_ms:.3f}",
                ]
            )


def analyze(
    root: str,
    folders: Sequence[str],
    reference: str,
    out_dir: str,
    recursive: bool,
    max_match_ms: float,
) -> dict:
    """主分析流程。"""

    folder_paths = {name: os.path.join(root, name) for name in folders}

    timelines: dict[str, List[ImageStamp]] = {}
    for name, path in folder_paths.items():
        if not os.path.isdir(path):
            raise FileNotFoundError(f"目录不存在: {path}")
        timelines[name] = load_timeline(path, recursive=recursive)

    # 统一基准时间，避免 datetime.timestamp 的时区细节。
    all_ts = [x.ts for items in timelines.values() for x in items]
    if not all_ts:
        raise RuntimeError("未找到任何图片文件。")

    base = min(all_ts)

    ensure_dir(out_dir)

    per_folder: dict[str, dict[str, object]] = {}
    pairwise: dict[str, dict[str, object]] = {}

    summary: dict[str, object] = {
        "root": os.path.abspath(root),
        "folders": list(folders),
        "reference": reference,
        "max_match_ms": max_match_ms,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "per_folder": per_folder,
        "pairwise": pairwise,
    }

    # 1) 每路帧间隔统计
    for name in folders:
        tl = timelines[name]
        intervals_ms = compute_intervals_ms(tl)

        stats: dict[str, object] = {
            "image_count": len(tl),
            "first_ts": tl[0].ts.isoformat(timespec="microseconds") if tl else None,
            "last_ts": tl[-1].ts.isoformat(timespec="microseconds") if tl else None,
        }

        if intervals_ms:
            s = sorted(intervals_ms)
            median_dt = median(s)

            # “延迟/卡顿”通常表现为 dt 远大于正常帧间隔。这里用 4*median 作为默认阈值。
            # 如果 median 很小（高帧率），至少用 1000ms 做一个保底阈值，避免误报。
            gap_threshold_ms = max(4.0 * median_dt, 1000.0)
            gap_count = sum(1 for v in intervals_ms if v > gap_threshold_ms)
            gap_max_ms = max(intervals_ms) if intervals_ms else 0.0

            stats.update(
                {
                    "dt_median_ms": median_dt,
                    "fps_estimate": (1000.0 / median_dt) if median_dt > 0 else None,
                    "dt_p95_ms": _percentile(s, 95),
                    "dt_p99_ms": _percentile(s, 99),
                    "dt_max_ms": s[-1],
                    "gap_threshold_ms": gap_threshold_ms,
                    "gap_count": gap_count,
                    "gap_max_ms": gap_max_ms,
                }
            )

        per_folder[name] = stats

        write_intervals_csv(
            os.path.join(out_dir, f"intervals_{name}.csv"), intervals_ms
        )

    # 2) 与参考相机的同步分析
    if reference not in timelines:
        raise ValueError(f"reference 不在 folders 中: {reference}")

    ref_tl = timelines[reference]

    for name in folders:
        if name == reference:
            continue

        pairs = match_monotonic_nearest(
            ref=ref_tl,
            cam=timelines[name],
            max_abs_diff_ms=max_match_ms,
            base=base,
        )

        deltas_ms = [p.delta_us / 1000.0 for p in pairs]
        if deltas_ms:
            offset_ms = median(deltas_ms)
        else:
            offset_ms = float("nan")

        residuals_ms = [d - offset_ms for d in deltas_ms] if deltas_ms else []
        abs_residuals_ms = [abs(r) for r in residuals_ms]

        # 漂移：x=ref_time(s), y=delta(ms)
        x_s = [(_to_us(p.ref.ts, base) / 1_000_000.0) for p in pairs]
        y_ms = deltas_ms
        slope_ms_per_s = _linear_regression_slope(x_s, y_ms)

        key = f"{name}_to_{reference}"
        pairwise[key] = {
            "matched_pairs": len(pairs),
            "matched_ratio_ref": (len(pairs) / len(ref_tl)) if ref_tl else 0.0,
            "matched_ratio_cam": (len(pairs) / len(timelines[name])) if timelines[name] else 0.0,
            "offset_median_ms": offset_ms,
            "delta_stats": summarize_deltas_ms(deltas_ms),
            "residual_stats": summarize_deltas_ms(residuals_ms),
            "abs_residual_stats": summarize_deltas_ms(abs_residuals_ms),
            "drift_slope_ms_per_s": slope_ms_per_s,
            "drift_ms_per_min": slope_ms_per_s * 60.0 if slope_ms_per_s == slope_ms_per_s else None,
        }

        write_matches_csv(
            os.path.join(out_dir, f"matches_{name}_to_{reference}.csv"),
            pairs,
            offset_ms=offset_ms if offset_ms == offset_ms else 0.0,
        )

    # 3) 写汇总
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="三路相机图片时间同步与延迟分析")
    p.add_argument("--root", default=".", help="数据根目录（包含三路相机文件夹）")
    p.add_argument(
        "--folders",
        nargs="+",
        default=[
            "MV-CS050-04(DA8199243)",
            "MV-CS050-02(DA8199285)",
            "MV-CS050-03(DA8199303)",
            "MV-CS050-CTRL(DA8199402)",
        ],
        help="需要分析的文件夹名称（相对于 root）",
    )
    p.add_argument(
        "--reference",
        default="MV-CS050-03(DA8199303)",
        help="参考相机文件夹名称（必须在 --folders 中）",
    )
    p.add_argument(
        "--out",
        default="data/tools_output",
        help="输出目录（相对于当前工作目录）",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子目录（默认只扫第一层）",
    )
    p.add_argument(
        "--max-match-ms",
        type=float,
        default=80.0,
        help="匹配窗口：允许的最大绝对时间差(毫秒)",
    )
    return p


def main() -> None:
    # Windows + Git Bash 下将输出重定向到文件时，容易因代码页导致中文乱码。
    # 这里强制 stdout/stderr 用 UTF-8，便于在 output.txt 中直接查看。
    try:
        for stream in (sys.stdout, sys.stderr):
            reconfig = getattr(stream, "reconfigure", None)
            if callable(reconfig):
                reconfig(encoding="utf-8")
    except Exception:
        pass

    args = build_arg_parser().parse_args()

    summary = analyze(
        root=args.root,
        folders=args.folders,
        reference=args.reference,
        out_dir=args.out,
        recursive=args.recursive,
        max_match_ms=args.max_match_ms,
    )

    # 控制台输出精简摘要（便于快速判断是否同步）
    print("=== 图片时间同步分析完成 ===")
    print(f"输出目录: {os.path.abspath(args.out)}")
    print(f"参考相机: {args.reference}")
    for name in args.folders:
        info = summary["per_folder"][name]
        print(
            f"[{name}] count={info.get('image_count')} "
            f"dt_median_ms={info.get('dt_median_ms')} fps≈{info.get('fps_estimate')}"
        )

    for k, v in summary["pairwise"].items():
        print(
            f"[{k}] matched={v.get('matched_pairs')} "
            f"offset_median_ms={v.get('offset_median_ms')} "
            f"residual_p95_ms={v.get('residual_stats', {}).get('p95_ms')} "
            f"drift_ms_per_min={v.get('drift_ms_per_min')}"
        )


if __name__ == "__main__":
    main()
