"""融合（多目定位）输出的类型协议。

用途：
    该模块刻意只依赖 Python 标准库 typing，便于跨项目复用。

背景：
    本仓库的 `tennis3d.pipeline.core.run_localization_pipeline()` 会输出可 JSON
    序列化的 dict（通常会被写入 jsonl）。其中每条记录包含 0..N 个球的融合结果。

注意：
    - 输出记录里除了固定字段外，可能还会透传上游 meta（例如 frame_id、时间戳等）。
      因此这里把 record 的“额外字段”留给调用方处理。
    - 协方差的单位：
        - 像素域：px^2（2x2）
        - 世界坐标 3D：m^2（3x3）
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, NotRequired, Protocol, TypedDict


class ReprojectionError(TypedDict):
    """单相机重投影误差诊断。"""

    camera: str
    uv: list[float]  # [u, v]
    uv_hat: list[float]  # [u_hat, v_hat]
    error_px: float


class DetectionDetails(TypedDict):
    """每相机被选中 detection 的详细信息（Optional输出）。"""

    bbox: list[float]  # [x1, y1, x2, y2]
    score: float
    cls: int
    center: list[float]  # [u, v]


class Obs2DWithCov(TypedDict):
    """每相机 2D 观测与协方差。

    说明：
        - 该结构用于后续更强的轨迹拟合（像素域重投影误差最小化、加权、鲁棒等）。
        - cov_uv 通常为 2x2 对称阵，单位 px^2。
    """

    uv: list[float]  # [u, v]
    cov_uv: list[list[float]]  # 2x2
    sigma_px: float
    cov_source: str

    # 以下诊断字段：可能缺失或为 None。
    uv_hat: NotRequired[list[float] | None]
    reproj_error_px: NotRequired[float | None]
    detection_index: NotRequired[int | None]


class TriangulationStats(TypedDict):
    """三角化几何统计（用于判断视角退化）。"""

    num_pairs: int
    ray_angle_deg_min: float | None
    ray_angle_deg_median: float | None
    ray_angle_deg_max: float | None


class FusedBall(TypedDict):
    """单个球的融合输出（多目定位结果）。"""

    ball_id: int

    # 主要输出：世界坐标 3D 点（米）
    ball_3d_world: list[float]  # [x, y, z]

    # Optional增强：每相机坐标系下的 3D 点（米），用于诊断/可视化。
    ball_3d_camera: dict[str, list[float]]

    used_cameras: list[str]
    num_views: int

    # 点级质量（无量纲）。在本仓库中用于下游的加权/排序。
    quality: float

    # 重投影误差统计（像素）
    median_reproj_error_px: float
    max_reproj_error_px: float
    reprojection_errors: list[ReprojectionError]

    # Optional：每相机用于该 ball 的 detection 详情（若上游开启 include_detection_details）。
    detections: NotRequired[dict[str, DetectionDetails]]

    # 新增：每相机 2D 观测与协方差。
    obs_2d_by_camera: NotRequired[dict[str, Obs2DWithCov]]

    # 新增：3D 点协方差（世界坐标，m^2）。不可估计时可能为 None。
    ball_3d_cov_world: NotRequired[list[list[float]] | None]  # 3x3

    # 新增：3D 点标准差（米），便于人读诊断。
    ball_3d_std_m: NotRequired[list[float] | None]  # [sx, sy, sz]

    # 新增：三角化几何统计。
    triangulation_stats: NotRequired[TriangulationStats]


class FusedLocalizationRecord(TypedDict):
    """一帧/一组的融合输出记录。"""

    created_at: float
    balls: list[FusedBall]

    # 其它 meta 字段（例如 frame_id、capture_t_abs、序列号等）会被透传，
    # 由于项目差异较大，这里不做强约束。


class LocalizationRecordProducer(Protocol):
    """融合输出记录的生产者协议（例如 pipeline 生成器、消息订阅器等）。"""

    def __iter__(self) -> Iterator[FusedLocalizationRecord]:
        ...


class LocalizationRecordPostProcessor(Protocol):
    """融合记录的后处理协议（例如轨迹拟合 stage）。"""

    def process_record(self, out_rec: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


def iter_records(x: Iterable[FusedLocalizationRecord]) -> Iterator[FusedLocalizationRecord]:
    """一个小工具：把 Iterable 规范成 Iterator，便于类型推断。"""

    yield from x
