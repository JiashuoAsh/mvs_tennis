from __future__ import annotations

from tennis3d.sync.time_mapping_report import build_time_mapping_report


def test_build_time_mapping_report_summarizes_spread_and_deltas() -> None:
    # 说明：nearest-rank 分位数在 n=2 时：p50 取最小值，p95 取最大值。
    records = [
        {
            "time_mapping_host_ms_spread_ms": 10.0,
            "time_mapping_mapped_host_ms_spread_ms": 0.4,
            "time_mapping_mapped_host_ms_delta_to_median_by_camera": {"A": -0.2, "B": 0.2},
        },
        {
            "time_mapping_host_ms_spread_ms": 20.0,
            "time_mapping_mapped_host_ms_spread_ms": 0.1,
            "time_mapping_mapped_host_ms_delta_to_median_by_camera": {"A": -0.1, "B": 0.1},
        },
    ]

    r = build_time_mapping_report(records=records)

    assert r.groups_used == 2

    assert r.raw_host_spread_ms is not None
    assert r.raw_host_spread_ms.p50 == 10.0
    assert r.raw_host_spread_ms.p95 == 20.0
    assert r.raw_host_spread_ms.max == 20.0

    assert r.mapped_spread_ms is not None
    assert r.mapped_spread_ms.p50 == 0.1
    assert r.mapped_spread_ms.p95 == 0.4
    assert r.mapped_spread_ms.max == 0.4

    assert set(r.mapped_delta_by_camera.keys()) == {"A", "B"}

    # n=2：median 取最小（更贴近 nearest-rank 的定义）
    assert r.mapped_delta_by_camera["A"].median == -0.2
    assert r.mapped_delta_by_camera["A"].abs_p95 == 0.2
    assert r.mapped_delta_by_camera["B"].median == 0.1
    assert r.mapped_delta_by_camera["B"].abs_p95 == 0.2


def test_build_time_mapping_report_falls_back_to_mapped_host_ms() -> None:
    records = [
        {
            "time_mapping_mapped_host_ms_by_camera": {"A": 1.0, "B": 2.0},
            "time_mapping_mapped_host_ms_spread_ms": 1.0,
        }
    ]

    r = build_time_mapping_report(records=records)
    assert r.groups_used == 1

    assert r.mapped_spread_ms is not None
    assert r.mapped_spread_ms.p50 == 1.0
    assert r.mapped_spread_ms.p95 == 1.0
    assert r.mapped_spread_ms.max == 1.0

    # 偶数个元素时取上中位数：median=2.0
    # delta: A=-1, B=0
    assert r.mapped_delta_by_camera["A"].median == -1.0
    assert r.mapped_delta_by_camera["B"].median == 0.0
