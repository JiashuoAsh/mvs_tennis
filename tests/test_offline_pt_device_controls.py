from __future__ import annotations

import json
from pathlib import Path

from tennis3d.apps.offline_localize_from_captures import build_arg_parser
from tennis3d.config import load_offline_app_config


def test_offline_build_arg_parser_accepts_pt_device() -> None:
    # 说明：该测试不读取 captures，也不加载模型，仅验证 CLI 参数解析。
    p = build_arg_parser()
    args = p.parse_args(
        [
            "--captures-dir",
            "data/captures_master_slave/tennis_test",
            "--calib",
            "data/calibration/example_triple_camera_calib.json",
            "--detector",
            "pt",
            "--model",
            "data/models/best.pt",
            "--pt-device",
            "cuda:0",
        ]
    )

    assert str(getattr(args, "pt_device")) == "cuda:0"


def test_load_offline_app_config_supports_pt_device(tmp_path: Path) -> None:
    # 说明：使用 JSON 配置来避免测试环境对 PyYAML 的依赖。
    cfg_path = tmp_path / "offline.json"
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
                    "pt_device": "cuda:0",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = load_offline_app_config(cfg_path)
    assert cfg.detector == "pt"
    assert cfg.pt_device == "cuda:0"
