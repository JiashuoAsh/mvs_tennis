# Copilot 会话日志（本地 tmp）

- 会话开始时间：2026-02-06T00:00:00+08:00

---

## 2026-02-06T00:00:00+08:00

- 用户诉求：online 过程中希望能实时统计每个 loop 内各事件耗时。
- Copilot 输出：
  - 发现 pipeline 已在 out_rec['latency_host'] 提供 align/detect/localize/total 与每相机 detect 耗时。
  - 新增可选开关 terminal_timing（CLI: --terminal-timing；配置：output.terminal_timing）。
  - 在输出循环中按 record 打印 timing 行：包含 loop_last、pipe_ms 分解、out_ms(write/print) 与 det_cam_ms。
  - 默认关闭，避免改变既有终端输出与性能。
- 涉及文件：
  - packages/tennis3d_core/src/tennis3d/config.py
  - packages/tennis3d_online/src/tennis3d_online/cli.py
  - packages/tennis3d_online/src/tennis3d_online/spec.py
  - packages/tennis3d_online/src/tennis3d_online/output_loop.py
  - tests/test_online_output_controls.py
- 验证方式：运行全量测试，全部通过。

---

## 2026-02-06T00:00:00+08:00

- 用户诉求：把新增的逐 loop 耗时统计开关同步到实际 config 与 templates。
- Copilot 输出：
  - 在真实配置 `configs/online/master_slave_line0.yaml` 的 output 段加入 `terminal_timing: true` 与使用建议。
  - 更新模板 `examples/configs/templates/online_master_slave_template.yaml`：补齐 output 控制字段（JSONL flush、status 心跳、terminal_timing）。
  - 更新模板 `examples/configs/templates/online_software_trigger_minimal.yaml`：加入 `terminal_timing` 字段与注释。
- 涉及文件：
  - configs/online/master_slave_line0.yaml
  - examples/configs/templates/online_master_slave_template.yaml
  - examples/configs/templates/online_software_trigger_minimal.yaml
- 验证方式：运行全量测试，全部通过。
