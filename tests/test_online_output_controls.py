from __future__ import annotations

import json
from pathlib import Path

from tennis3d_online.cli import build_arg_parser
from tennis3d_online.spec import build_spec_from_config
from tennis3d.config import load_online_app_config


def test_build_arg_parser_accepts_terminal_print_mode_none_and_jsonl_flush_flags() -> None:
    # 说明：该测试不连接相机，仅验证 CLI 参数解析。
    p = build_arg_parser()
    args = p.parse_args(
        [
            "--serial",
            "A",
            "--pt-device",
            "cuda:0",
            "--terminal-print-mode",
            "none",
            "--terminal-status-interval-s",
            "1.0",
            "--terminal-timing",
            "--out-jsonl",
            "data/tools_output/x.jsonl",
            "--out-jsonl-only-when-balls",
            "--out-jsonl-flush-every-records",
            "10",
            "--out-jsonl-flush-interval-s",
            "0.5",
        ]
    )

    assert str(getattr(args, "pt_device")) == "cuda:0"
    assert str(args.terminal_print_mode) == "none"
    assert float(args.terminal_status_interval_s) == 1.0
    assert bool(getattr(args, "terminal_timing", False)) is True
    assert str(args.out_jsonl).endswith("x.jsonl")
    assert bool(args.out_jsonl_only_when_balls) is True
    assert int(args.out_jsonl_flush_every_records) == 10
    assert float(args.out_jsonl_flush_interval_s) == 0.5


def test_load_online_app_config_supports_output_controls(tmp_path: Path) -> None:
    # 说明：使用 JSON 配置来避免测试环境对 PyYAML 的依赖。
    cfg_path = tmp_path / "online.json"
    cfg_path.write_text(
        json.dumps(
            {
                "sdk": {
                    "mvimport_dir": "",
                    "dll_dir": "",
                },
                "camera": {
                    "serials": ["A", "B"],
                    "calib": "data/calibration/example_triple_camera_calib.json",
                    "exposure": {"auto": "Off", "time_us": 8000.0},
                    "gain": {"auto": "Off", "value": 6.0},
                },
                "detector": {
                    "pt_device": "cuda:0",
                },
                "output": {
                    "terminal_print_mode": "none",
                    "terminal_status_interval_s": 1.0,
                    "terminal_timing": True,
                    "out_jsonl": "data/tools_output/x.jsonl",
                    "out_jsonl_only_when_balls": True,
                    "out_jsonl_flush_every_records": 10,
                    "out_jsonl_flush_interval_s": 0.5,
                },
                "trigger": {
                    "trigger_source": "Software",
                    "master_serial": "",
                    "soft_trigger_fps": 5.0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = load_online_app_config(cfg_path)
    assert cfg.pt_device == "cuda:0"
    assert cfg.terminal_print_mode == "none"
    assert cfg.terminal_status_interval_s == 1.0
    assert bool(getattr(cfg, "terminal_timing", False)) is True
    assert cfg.out_jsonl is not None
    assert Path(cfg.out_jsonl).as_posix().endswith("data/tools_output/x.jsonl")
    assert cfg.out_jsonl_only_when_balls is True
    assert cfg.out_jsonl_flush_every_records == 10
    assert cfg.out_jsonl_flush_interval_s == 0.5

    assert cfg.exposure_auto == "Off"
    assert float(cfg.exposure_time_us or 0.0) == 8000.0
    assert cfg.gain_auto == "Off"
    assert float(cfg.gain or 0.0) == 6.0

    # 说明：spec 是 entry 与 runtime 的稳定边界；这里验证映射不会丢字段。
    spec = build_spec_from_config(cfg)
    assert spec.exposure_auto == "Off"
    assert float(spec.exposure_time_us or 0.0) == 8000.0
    assert spec.gain_auto == "Off"
    assert float(spec.gain or 0.0) == 6.0
    assert bool(getattr(spec, "terminal_timing", False)) is True
