# -*- coding: utf-8 -*-

"""运行中动态 ROI 探测工具（MVS / GenICam）。

目的：
- 回答一个非常现实的问题：你的相机 + MVS 驱动是否支持在采集开始后（StartGrabbing）高频修改 ROI？
- 尤其关注 OffsetX/OffsetY 的运行时可写性（动态平移 ROI）。

背景（来自本仓库的工程经验）：
- `src/mvs/sdk/camera.py:configure_resolution()` 与 `configure_pixel_format()` 都明确建议：
  “建议在 StartGrabbing 之前设置（采集开始后很多节点会被锁定为不可写）”。
- 但“建议”不等于“绝对不行”：具体是否可写，取决于机型/固件/当前模式（触发/连续采集）等。

本脚本做的事：
1) 打开指定序列号的相机并启动取流（StartGrabbing）。
2) （可选）启用软触发，以便让相机持续产出帧（更贴近真实在线场景）。
3) 在运行中循环调用 MV_CC_SetIntValue("OffsetX"/"OffsetY")，统计：
   - 成功率（ret == MV_OK）
   - 失败返回码直方图（例如 MV_E_ACCESS_DENIED）
   - 每次调用耗时（粗略）

注意：
- 本工具只探测“能不能写”。即使能写，高频修改 ROI 仍可能带来：
  - 丢帧 / 延迟抖动
  - 多相机同步破坏（master/slave 更敏感）
  - Width/Height/Offset 的步进约束导致 ROI 不连续

建议：
- 若你的目标是“每帧跟随球位置”那种频率，工程上更可靠的是：保持相机输出不变，做软件裁剪（本仓库已实现）。
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter

from mvs import MvsCamera, MvsSdk, SoftwareTriggerLoop, enumerate_devices, load_mvs_binding


def _get_int_node_info(*, binding, cam, key: str) -> tuple[int, int, int, int] | None:
    """读取 int 节点的当前值/最小/最大/步进。

    说明：
        - 这里重复实现一份轻量 helper，避免在工具脚本中依赖 `mvs.sdk.camera` 的私有函数。
    """

    try:
        st = binding.params.MVCC_INTVALUE()
        ret = int(cam.MV_CC_GetIntValue(str(key), st))
        if ret != int(binding.MV_OK):
            return None
        cur = int(getattr(st, "nCurValue"))
        vmin = int(getattr(st, "nMin"))
        vmax = int(getattr(st, "nMax"))
        inc = int(getattr(st, "nInc"))
        return cur, vmin, vmax, inc
    except Exception:
        return None


def _warmup_genicam_xml(*, binding, cam) -> bool:
    """尝试预热/生成 GenICam XML。

    说明：
        - 某些环境下，XML 相关接口可能需要先触发一次 MV_XML_GetGenICamXML。
        - 该函数 best-effort：失败不抛异常，返回 False。
    """

    try:
        import ctypes

        # 先用 NULL buffer 获取 XML 长度。
        xml_len = ctypes.c_uint(0)
        null_buf = ctypes.c_void_p(0)

        ret = int(cam.MV_XML_GetGenICamXML(null_buf, 0, xml_len))
        if ret != int(binding.MV_OK):
            return False
        n = int(getattr(xml_len, "value", 0))
        if n <= 0:
            return False

        buf = ctypes.create_string_buffer(n)
        ret2 = int(cam.MV_XML_GetGenICamXML(ctypes.cast(buf, ctypes.c_void_p), int(n), xml_len))
        return int(ret2) == int(binding.MV_OK)
    except Exception:
        return False


def _get_access_mode(*, binding, cam, node_name: str) -> tuple[int | None, int | None]:
    """读取节点访问模式（GenICam XML Access Mode）。

    说明：
        - 官方 SDK 暴露了 MV_XML_GetNodeAccessMode 接口，可直接查询节点在“当前状态”下是否可读写。
        - 这比“直接 SetIntValue 看报错”更快更可解释：
          如果取流后 access mode 变成 AM_RO/AM_NA/AM_NI，那么高频动态 ROI 基本就不用想了。
    """

    try:
        mode = binding.params.MV_XML_AccessMode()
        ret = int(cam.MV_XML_GetNodeAccessMode(str(node_name), mode))
        if ret != int(binding.MV_OK):
            return None, int(ret)
        return int(getattr(mode, "value", int(mode))), int(ret)
    except Exception:
        return None, None


def _format_access_mode(*, binding, mode: int | None) -> str:
    if mode is None:
        return "(unknown)"

    names = {
        int(binding.params.AM_NI): "AM_NI(Not implemented)",
        int(binding.params.AM_NA): "AM_NA(Not available)",
        int(binding.params.AM_WO): "AM_WO(Write only)",
        int(binding.params.AM_RO): "AM_RO(Read only)",
        int(binding.params.AM_RW): "AM_RW(Read/Write)",
        int(binding.params.AM_Undefined): "AM_Undefined",
        int(binding.params.AM_CycleDetect): "AM_CycleDetect",
    }
    return names.get(int(mode), f"(unknown:{int(mode)})")


def _align_down(value: int, *, vmin: int, vmax: int, inc: int) -> int:
    if inc <= 0:
        inc = 1
    v = int(value)
    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax
    steps = (v - vmin) // inc
    return int(vmin + steps * inc)


def _find_device_index(*, binding, serial: str) -> tuple[object, int, int]:
    st_dev_list, descs = enumerate_devices(binding)
    for d in descs:
        if str(d.serial) == str(serial):
            return st_dev_list, int(d.index), int(d.tlayer_type)
    raise RuntimeError(f"未找到 serial={serial} 的设备（请先运行：python -m mvs.apps.quad_capture --list）")


def main() -> int:
    ap = argparse.ArgumentParser(description="探测 MVS 相机是否支持运行中高频修改 ROI OffsetX/OffsetY")
    ap.add_argument("--serial", required=True, help="相机序列号")

    ap.add_argument("--roi-width", type=int, default=1280, help="启动取流前设置的 ROI 宽度（仅用于给 Offset 留空间）")
    ap.add_argument("--roi-height", type=int, default=1080, help="启动取流前设置的 ROI 高度（仅用于给 Offset 留空间）")

    ap.add_argument("--set-hz", type=float, default=30.0, help="Offset 设置频率（Hz）")
    ap.add_argument("--iters", type=int, default=300, help="尝试次数")

    ap.add_argument("--step-px", type=int, default=64, help="Offset 步进（像素）；会按节点 inc 向下对齐")
    ap.add_argument("--y", type=int, default=0, help="OffsetY 固定值（会按 inc 对齐）")

    ap.add_argument("--enable-soft-trigger-fps", type=float, default=0.0, help=">0 时启用软触发（更贴近在线出图场景）")

    args = ap.parse_args()

    binding = load_mvs_binding()

    with MvsSdk(binding) as _sdk:
        st_dev_list, dev_index, tlayer_type = _find_device_index(binding=binding, serial=str(args.serial))

        # 用 Software 触发打开：无需外部接线也能启动取流。
        # 注意：是否产出帧由 soft trigger 决定（若不开启，仍可测试“节点是否可写”。）
        with MvsCamera.open_from_device_list(
            binding=binding,
            st_dev_list=st_dev_list,
            dev_index=int(dev_index),
            serial=str(args.serial),
            tlayer_type=int(tlayer_type),
            trigger_source="Software",
            trigger_activation="FallingEdge",
            trigger_cache_enable=True,
            pixel_format="",
            image_width=int(args.roi_width),
            image_height=int(args.roi_height),
            image_offset_x=0,
            image_offset_y=0,
            exposure_auto="",
            exposure_time_us=None,
            gain_auto="",
            gain=None,
        ) as cam:
            trig_stop = None
            trig = None
            if float(args.enable_soft_trigger_fps) > 0:
                import threading

                trig_stop = threading.Event()
                trig = SoftwareTriggerLoop(
                    targets=[(str(args.serial), cam.cam)],
                    stop_event=trig_stop,
                    fps=float(args.enable_soft_trigger_fps),
                    out_q=None,
                )
                trig.start()

            try:
                # 读取 OffsetX/OffsetY 的范围与步进。
                # 同时读取节点访问模式：这是官方 SDK 提供的“当前节点是否可读写”的权威答案。
                # 由于本脚本的相机已进入取流（StartGrabbing），这里得到的是“运行中”的 access mode。
                ox_am, ox_am_ret = _get_access_mode(binding=binding, cam=cam.cam, node_name="OffsetX")
                oy_am, oy_am_ret = _get_access_mode(binding=binding, cam=cam.cam, node_name="OffsetY")
                w_am, w_am_ret = _get_access_mode(binding=binding, cam=cam.cam, node_name="Width")
                h_am, h_am_ret = _get_access_mode(binding=binding, cam=cam.cam, node_name="Height")

                # best-effort：如果 access mode 查询失败，尝试预热 XML 后再查一次。
                # （部分环境下 XML 相关接口可能需要先触发一次 MV_XML_GetGenICamXML。）
                if (
                    (ox_am is None and ox_am_ret is not None and ox_am_ret != int(binding.MV_OK))
                    or (oy_am is None and oy_am_ret is not None and oy_am_ret != int(binding.MV_OK))
                    or (w_am is None and w_am_ret is not None and w_am_ret != int(binding.MV_OK))
                    or (h_am is None and h_am_ret is not None and h_am_ret != int(binding.MV_OK))
                ):
                    if _warmup_genicam_xml(binding=binding, cam=cam.cam):
                        ox_am, ox_am_ret = _get_access_mode(binding=binding, cam=cam.cam, node_name="OffsetX")
                        oy_am, oy_am_ret = _get_access_mode(binding=binding, cam=cam.cam, node_name="OffsetY")
                        w_am, w_am_ret = _get_access_mode(binding=binding, cam=cam.cam, node_name="Width")
                        h_am, h_am_ret = _get_access_mode(binding=binding, cam=cam.cam, node_name="Height")

                ox_info = _get_int_node_info(binding=binding, cam=cam.cam, key="OffsetX")
                oy_info = _get_int_node_info(binding=binding, cam=cam.cam, key="OffsetY")
                if ox_info is None or oy_info is None:
                    raise RuntimeError("无法读取 OffsetX/OffsetY 节点信息：该机型可能不支持 ROI，或当前节点不可读。")

                _ox_cur, ox_min, ox_max, ox_inc = ox_info
                _oy_cur, oy_min, oy_max, oy_inc = oy_info

                y_fixed = _align_down(int(args.y), vmin=int(oy_min), vmax=int(oy_max), inc=int(oy_inc))

                period = 1.0 / max(float(args.set_hz), 0.1)
                dts_ms: list[float] = []
                rets = Counter()

                last_set_x: int | None = None
                last_set_y: int | None = None

                next_t = time.perf_counter()
                for i in range(int(args.iters)):
                    now = time.perf_counter()
                    if now < next_t:
                        time.sleep(min(0.005, next_t - now))

                    # 在 [ox_min, ox_max] 内做一个来回扫描，避免只测到单点。
                    span = max(0, int(ox_max) - int(ox_min))
                    step = max(1, int(args.step_px))
                    if span <= 0:
                        x_raw = int(ox_min)
                    else:
                        # sawtooth: 0..span..0
                        phase = int(i * step) % (2 * span if span > 0 else 1)
                        x_raw = int(ox_min) + (phase if phase <= span else (2 * span - phase))

                    x = _align_down(int(x_raw), vmin=int(ox_min), vmax=int(ox_max), inc=int(ox_inc))

                    t0 = time.perf_counter()
                    ret_x = int(cam.cam.MV_CC_SetIntValue("OffsetX", int(x)))
                    ret_y = int(cam.cam.MV_CC_SetIntValue("OffsetY", int(y_fixed)))
                    t1 = time.perf_counter()

                    last_set_x = int(x)
                    last_set_y = int(y_fixed)

                    # 记录“较差的那个返回码”，用于快速判断是否被锁。
                    ret = ret_x if ret_x != int(binding.MV_OK) else ret_y
                    rets[int(ret)] += 1
                    dts_ms.append(float((t1 - t0) * 1000.0))

                    next_t += period

                ok = int(rets.get(int(binding.MV_OK), 0))
                total = int(sum(rets.values()))
                ok_rate = (float(ok) / float(total)) if total > 0 else 0.0

                # 输出结果：
                print("=== MVS runtime ROI probe ===")
                print(f"serial={args.serial}")
                print("node access mode (StartGrabbing 后):")
                ox_ret_str = f"0x{int(ox_am_ret):08X}" if ox_am_ret is not None else "(unknown)"
                oy_ret_str = f"0x{int(oy_am_ret):08X}" if oy_am_ret is not None else "(unknown)"
                w_ret_str = f"0x{int(w_am_ret):08X}" if w_am_ret is not None else "(unknown)"
                h_ret_str = f"0x{int(h_am_ret):08X}" if h_am_ret is not None else "(unknown)"

                print(f"  OffsetX: {_format_access_mode(binding=binding, mode=ox_am)} (ret={ox_ret_str})")
                print(f"  OffsetY: {_format_access_mode(binding=binding, mode=oy_am)} (ret={oy_ret_str})")
                print(f"  Width:   {_format_access_mode(binding=binding, mode=w_am)} (ret={w_ret_str})")
                print(f"  Height:  {_format_access_mode(binding=binding, mode=h_am)} (ret={h_ret_str})")
                print(f"StartGrabbing 后写 OffsetX/OffsetY：iters={total}, ok={ok} ({ok_rate:.3f})")
                print(f"OffsetX range=[{ox_min}, {ox_max}] inc={ox_inc}; OffsetY range=[{oy_min}, {oy_max}] inc={oy_inc}")
                print(f"OffsetY fixed={y_fixed}")

                # 读回验证：确认节点值确实变化到最后一次设置的值。
                # 注意：这只验证“节点值”，不保证每一帧都严格按新 ROI 出图（多相机同步场景仍需另外观测）。
                if last_set_x is not None and last_set_y is not None:
                    ox_now = _get_int_node_info(binding=binding, cam=cam.cam, key="OffsetX")
                    oy_now = _get_int_node_info(binding=binding, cam=cam.cam, key="OffsetY")
                    if ox_now is not None and oy_now is not None:
                        print(
                            "readback: "
                            f"OffsetX cur={ox_now[0]} (last_set={last_set_x}); "
                            f"OffsetY cur={oy_now[0]} (last_set={last_set_y})"
                        )

                if dts_ms:
                    p50 = statistics.median(dts_ms)
                    p95 = statistics.quantiles(dts_ms, n=20)[18]  # 95% 近似
                    print(f"SetIntValue call time (ms): median≈{p50:.3f}, p95≈{p95:.3f}, max={max(dts_ms):.3f}")

                print("ret histogram (top):")
                for ret, cnt in rets.most_common(8):
                    print(f"  ret=0x{int(ret):08X} count={int(cnt)}")

                # 粗略解释：如果出现 ACCESS_DENIED，通常就是运行中被锁。
                try:
                    denied = int(binding.err.MV_E_ACCESS_DENIED)
                except Exception:
                    denied = None
                if denied is not None and int(rets.get(int(denied), 0)) > 0:
                    print("\n结论提示：检测到 MV_E_ACCESS_DENIED，说明运行中修改 Offset 节点很可能被锁定。")

                # 如果 access mode 明确不是可写，给一个更“提前”的提示。
                try:
                    am_rw = int(binding.params.AM_RW)
                    am_wo = int(binding.params.AM_WO)
                except Exception:
                    am_rw = None
                    am_wo = None

                if am_rw is not None and am_wo is not None:
                    if ox_am is not None and oy_am is not None and (ox_am not in (am_rw, am_wo) or oy_am not in (am_rw, am_wo)):
                        print(
                            "\n结论提示：OffsetX/OffsetY 的访问模式不是可写（AM_RW/AM_WO）。"
                            "这通常意味着‘取流后不支持动态 ROI’，需要停采/重启采集才能改，"
                            "或改走软件裁剪方案。"
                        )
                    elif ox_am is None or oy_am is None:
                        print(
                            "\n提示：未能通过 MV_XML_GetNodeAccessMode 读取 OffsetX/OffsetY 的访问模式。"
                            "这不影响 SetIntValue 压测结论；若 SetIntValue 持续返回 MV_OK，"
                            "就说明运行中写 Offset 至少在当前机型/模式下是可行的。"
                        )

                return 0
            finally:
                if trig_stop is not None:
                    trig_stop.set()
                if trig is not None:
                    trig.join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
