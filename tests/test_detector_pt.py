from __future__ import annotations

import json
from pathlib import Path

import pytest

from tennis3d.config import load_offline_app_config
from tennis3d_detectors import create_detector


def test_offline_config_accepts_pt(tmp_path: Path) -> None:
    """配置解析应当接受 detector=pt。"""

    cfg_path = tmp_path / "offline_cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "input": {
                    "captures_dir": "data/captures_master_slave/tennis_test",
                    "calib": "data/calibration/example_triple_camera_calib.json",
                },
                "detector": {
                    "name": "pt",
                    "model": "data/models/best.pt",
                    "min_score": 0.25,
                    "require_views": 2,
                },
                "run": {
                    "max_groups": 1,
                },
                "output": {
                    "out_jsonl": "data/tools_output/offline_positions_3d.jsonl",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = load_offline_app_config(cfg_path)
    assert cfg.detector == "pt"
    assert cfg.model is not None
    assert str(cfg.model).replace("\\", "/").endswith("data/models/best.pt")


def test_create_detector_pt_requires_model() -> None:
    with pytest.raises(ValueError, match=r"requires --model"):
        create_detector(name="pt", model_path=None, conf_thres=0.25)


def test_create_detector_unknown_message_includes_pt() -> None:
    with pytest.raises(ValueError) as ei:
        create_detector(name="something_else", model_path=None, conf_thres=0.25)
    assert "fake|color|rknn|pt" in str(ei.value)
