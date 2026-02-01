from __future__ import annotations

import json
from pathlib import Path

from tennis3d.config import load_offline_app_config
from tennis3d.trajectory.curve_stage import CurveStageConfig, apply_curve_stage


def _make_rec(*, t_abs: float, x: float, y: float, z: float) -> dict:
    # 说明：这里构造的是 run_localization_pipeline 的输出子集，足够覆盖 curve stage。
    return {
        "capture_t_abs": float(t_abs),
        "created_at": 123.0,
        "balls": [
            {
                "ball_id": 0,
                "ball_3d_world": [float(x), float(y), float(z)],
                "quality": 1.0,
            }
        ],
    }


def test_curve_stage_adds_curve_field_and_track_id() -> None:
    cfg = CurveStageConfig(
        enabled=True,
        max_tracks=2,
        association_dist_m=10.0,
        max_missed_s=10.0,
    )

    t0 = 1000.0
    records_in = [_make_rec(t_abs=t0 + i * 0.01, x=0.1 * i, y=1.5 - 0.2 * i, z=3.0) for i in range(6)]

    records_out = list(apply_curve_stage(records_in, cfg))
    assert len(records_out) == len(records_in)

    for r in records_out:
        assert "curve" in r
        assert r["curve"]["t_source"] == "capture_t_abs"
        assert int(r["curve"]["num_active_tracks"]) == 1

        balls = r.get("balls")
        assert isinstance(balls, list) and balls
        assert isinstance(balls[0], dict)
        assert balls[0].get("curve_track_id") == 1

    last = records_out[-1]
    tu = last["curve"]["track_updates"]
    assert isinstance(tu, list) and len(tu) == 1

    v3 = tu[0].get("v3")
    assert isinstance(v3, dict)
    assert v3.get("time_base_abs") == t0


def test_offline_config_accepts_curve_section(tmp_path: Path) -> None:
    cfg_path = tmp_path / "offline_cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "captures_dir": "data/captures_master_slave/tennis_test",
                "calib": "data/calibration/example_triple_camera_calib.json",
                "detector": "fake",
                "curve": {
                    "enabled": True,
                    "max_tracks": 2,
                    "conf_from": "constant",
                    "constant_conf": 0.5,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = load_offline_app_config(cfg_path)
    assert bool(cfg.curve.enabled) is True
    assert int(cfg.curve.max_tracks) == 2
    assert str(cfg.curve.conf_from) == "constant"
    assert float(cfg.curve.constant_conf) == 0.5
