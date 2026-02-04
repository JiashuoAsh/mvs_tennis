from __future__ import annotations

from tennis3d.pipeline.time_utils import delta_to_median_by_camera, host_timestamp_to_ms_epoch, spread_ms


def test_host_timestamp_to_ms_epoch_handles_ns_ms_s() -> None:
    # ns epoch
    assert host_timestamp_to_ms_epoch(1_700_000_000_000_000_000) == 1_700_000_000_000.0
    # ms epoch
    assert host_timestamp_to_ms_epoch(1_700_000_000_000) == 1_700_000_000_000.0
    # s epoch
    assert host_timestamp_to_ms_epoch(1_700_000_000) == 1_700_000_000_000.0


def test_host_timestamp_to_ms_epoch_returns_none_for_invalid() -> None:
    assert host_timestamp_to_ms_epoch(None) is None
    assert host_timestamp_to_ms_epoch("oops") is None
    assert host_timestamp_to_ms_epoch(-1) is None
    assert host_timestamp_to_ms_epoch(12345) is None


def test_spread_ms() -> None:
    assert spread_ms([1.0]) is None
    assert spread_ms([1.0, 4.0, 2.0]) == 3.0


def test_delta_to_median_by_camera_odd() -> None:
    d = delta_to_median_by_camera({"A": 100.0, "B": 101.0, "C": 99.0})
    assert d is not None
    assert d["A"] == 0.0
    assert d["B"] == 1.0
    assert d["C"] == -1.0


def test_delta_to_median_by_camera_even_uses_no_interpolation() -> None:
    # 说明：与 time_utils.median_float 一致：偶数个取上中位数（不做平均）。
    d = delta_to_median_by_camera({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    assert d is not None
    assert d["A"] == -2.0
    assert d["B"] == -1.0
    assert d["C"] == 0.0
    assert d["D"] == 1.0
