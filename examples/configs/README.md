# examples/configs：仅保留 templates

你现在应该把 `configs/` 当成“真正要用/可直接运行”的配置目录。

本目录只保留：

- `examples/configs/templates/`：**配置模板**（字段齐全、默认值尽量写出；必须填写的项用占位符）。

用法建议：

1) 从 `examples/configs/templates/*.yaml` 复制一份到 `configs/online/` 或 `configs/offline/`。
2) 把占位符（例如 `<CAM1_SERIAL>`、`<PATH_TO_CALIB>`）替换为你的真实值。
3) 用 `uv run ... --config configs/online/xxx.yaml`（或 `configs/offline/xxx.yaml`）运行。

