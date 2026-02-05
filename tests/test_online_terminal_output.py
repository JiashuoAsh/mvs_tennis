from __future__ import annotations

from tennis3d_online.terminal_format import (
    _format_all_balls_lines,
    _format_best_ball_line,
    _format_float3,
)


def test_format_float3() -> None:
    assert _format_float3([1, 2, 3]) == "(1.0000, 2.0000, 3.0000)"


def test_format_best_ball_line_returns_none_when_no_balls() -> None:
    assert _format_best_ball_line({"group_index": 1, "balls": []}) is None


def test_format_best_ball_line_formats_expected_fields() -> None:
    rec = {
        "group_index": 7,
        "capture_t_abs": 1.5,
        "balls": [
            {
                "ball_3d_world": [1.0, 2.0, 3.0],
                "used_cameras": ["A", "B"],
                "quality": 0.9,
                "num_views": 2,
                "reprojection_errors": [
                    {"camera": "A", "error_px": 1.0, "uv": [0.0, 0.0], "uv_hat": [0.0, 0.0]},
                    {"camera": "B", "error_px": 3.0, "uv": [0.0, 0.0], "uv_hat": [0.0, 0.0]},
                ],
            }
        ],
    }

    line = _format_best_ball_line(rec)
    assert line is not None
    assert "t=1.500000" in line
    assert "group=7" in line
    assert "xyz_w=(x=1.0000, y=2.0000, z=3.0000)" in line
    assert "q=0.900" in line
    assert "views=2" in line
    assert "err_mean=2.00px" in line
    assert "used=['A', 'B']" in line


def test_format_best_ball_line_includes_time_mapping_fields_when_present() -> None:
    rec = {
        "group_index": 1,
        "capture_t_abs": 2.0,
        "time_mapping_host_ms_spread_ms": 3.2,
        "time_mapping_mapped_host_ms_spread_ms": 0.4,
        "time_mapping_mapped_host_ms_delta_to_median_by_camera": {"A": -0.2, "B": 0.2},
        "balls": [
            {
                "ball_3d_world": [0.0, 0.0, 1.0],
                "used_cameras": ["A", "B"],
            }
        ],
    }

    line = _format_best_ball_line(rec)
    assert line is not None
    assert "dt_raw=3.200ms" in line
    assert "dt_map=0.400ms" in line
    assert "dt_map_by_cam={A:-0.200, B:+0.200}" in line


def test_format_all_balls_lines_returns_empty_when_no_balls() -> None:
    assert _format_all_balls_lines({"group_index": 1, "balls": []}) == []


def test_format_all_balls_lines_includes_header_and_each_ball() -> None:
    rec = {
        "group_index": 7,
        "capture_t_abs": 1.5,
        "balls": [
            {
                "ball_id": 0,
                "ball_3d_world": [1.0, 2.0, 3.0],
                "used_cameras": ["A", "B"],
                "quality": 0.9,
                "num_views": 2,
                "reprojection_errors": [
                    {"camera": "A", "error_px": 1.0, "uv": [0.0, 0.0], "uv_hat": [0.0, 0.0]},
                    {"camera": "B", "error_px": 3.0, "uv": [0.0, 0.0], "uv_hat": [0.0, 0.0]},
                ],
            },
            {
                "ball_id": 1,
                "ball_3d_world": [4.0, 5.0, 6.0],
                "used_cameras": ["A", "C"],
                "quality": 0.8,
                "num_views": 2,
                "median_reproj_error_px": 7.0,
            },
        ],
    }

    lines = _format_all_balls_lines(rec)
    assert len(lines) == 3
    assert lines[0].startswith("t=1.500000")
    assert "group=7" in lines[0]
    assert "balls=2" in lines[0]

    # ball 0
    assert "id=0" in lines[1]
    assert "xyz_w=(x=1.0000, y=2.0000, z=3.0000)" in lines[1]
    assert "q=0.900" in lines[1]
    assert "views=2" in lines[1]
    assert "err_mean=2.00px" in lines[1]
    assert "used=['A', 'B']" in lines[1]

    # ball 1
    assert "id=1" in lines[2]
    assert "xyz_w=(x=4.0000, y=5.0000, z=6.0000)" in lines[2]
    assert "q=0.800" in lines[2]
    assert "views=2" in lines[2]
    assert "err_mean=7.00px" in lines[2]
    assert "used=['A', 'C']" in lines[2]
