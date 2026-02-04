"""mvs.configure_resolution 的回归测试。

该测试不依赖真实相机或 MVS SDK，通过 fake cam/binding 模拟：
- 初始 OffsetX/OffsetY 非零会导致 WidthMax/HeightMax 被缩小；
- configure_resolution 应先清理 offset 再读取范围，从而能设置到期望的全分辨率。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from mvs import configure_resolution


class _FakeParams:
    class MVCC_INTVALUE:
        def __init__(self) -> None:
            self.nCurValue = 0
            self.nMin = 0
            self.nMax = 0
            self.nInc = 1


@dataclass
class _FakeBinding:
    """仅实现 configure_resolution 需要的最小 binding 接口。"""

    MV_OK: int = 0
    params: _FakeParams = _FakeParams()


class _FakeCam:
    """模拟 ROI 行为：Width/Height 的 max 受 OffsetX/OffsetY 影响。"""

    def __init__(self) -> None:
        self.sensor_w = 2448
        self.sensor_h = 2048

        # 模拟“历史配置残留”：offset 非零
        self.offset_x = 200
        self.offset_y = 400

        # 当前宽高也处于被裁剪状态
        self.width = self.sensor_w - self.offset_x
        self.height = self.sensor_h - self.offset_y

        self._width_inc = 16
        self._height_inc = 16

    def _width_max(self) -> int:
        return int(self.sensor_w - self.offset_x)

    def _height_max(self) -> int:
        return int(self.sensor_h - self.offset_y)

    def MV_CC_GetIntValue(self, key: str, st: Any) -> int:
        # 这里按 MVS 的字段名约定填充。
        if key == "Width":
            st.nCurValue = int(self.width)
            st.nMin = int(self._width_inc)
            st.nMax = int(self._width_max())
            st.nInc = int(self._width_inc)
            return 0
        if key == "Height":
            st.nCurValue = int(self.height)
            st.nMin = int(self._height_inc)
            st.nMax = int(self._height_max())
            st.nInc = int(self._height_inc)
            return 0
        if key == "OffsetX":
            st.nCurValue = int(self.offset_x)
            st.nMin = 0
            st.nMax = int(self.sensor_w - self._width_inc)
            st.nInc = int(self._width_inc)
            return 0
        if key == "OffsetY":
            st.nCurValue = int(self.offset_y)
            st.nMin = 0
            st.nMax = int(self.sensor_h - self._height_inc)
            st.nInc = int(self._height_inc)
            return 0

        return 1

    def MV_CC_SetIntValue(self, key: str, value: int) -> int:
        v = int(value)
        if key == "OffsetX":
            # 偏移范围/对齐约束这里简化处理。
            if v < 0:
                return 1
            self.offset_x = v
            # 宽高不能超过新的 max，否则相机通常会自动 clamp；这里用 clamp 模拟。
            self.width = min(int(self.width), self._width_max())
            return 0
        if key == "OffsetY":
            if v < 0:
                return 1
            self.offset_y = v
            self.height = min(int(self.height), self._height_max())
            return 0
        if key == "Width":
            if v <= 0:
                return 1
            if v > self._width_max():
                return 1
            self.width = v
            return 0
        if key == "Height":
            if v <= 0:
                return 1
            if v > self._height_max():
                return 1
            self.height = v
            return 0

        return 1


def test_configure_resolution_clears_offset_before_reading_range() -> None:
    binding = _FakeBinding()
    cam = _FakeCam()

    # 断言初始状态确实“被缩小”
    assert cam.offset_x == 200
    assert cam.offset_y == 400
    assert cam._width_max() == 2248
    assert cam._height_max() == 1648

    # 请求全分辨率 + offset 归零
    configure_resolution(
        binding=cast(Any, binding),
        cam=cam,
        width=2448,
        height=2048,
        offset_x=0,
        offset_y=0,
    )

    assert cam.offset_x == 0
    assert cam.offset_y == 0
    assert cam.width == 2448
    assert cam.height == 2048


def test_configure_resolution_rejects_non_positive_size() -> None:
    binding = _FakeBinding()
    cam = _FakeCam()

    with pytest.raises(ValueError):
        configure_resolution(binding=binding, cam=cam, width=0, height=2048)  # type: ignore[arg-type]


class _FixedAoiReadOnlySizeCam:
    """模拟“固定 AOI”机型：Width/Height 不可写，但 OffsetX/OffsetY 可写。"""

    def __init__(self) -> None:
        self.sensor_w = 2448
        self.sensor_h = 2048

        # 固定 AOI 尺寸（不可改）
        self.width = 1280
        self.height = 1080

        self.offset_x = 0
        self.offset_y = 0

        self._inc = 4

    def MV_CC_GetIntValue(self, key: str, st: Any) -> int:
        if key == "Width":
            st.nCurValue = int(self.width)
            st.nMin = int(self.width)
            st.nMax = int(self.width)
            st.nInc = int(self._inc)
            return 0
        if key == "Height":
            st.nCurValue = int(self.height)
            st.nMin = int(self.height)
            st.nMax = int(self.height)
            st.nInc = int(self._inc)
            return 0
        if key == "OffsetX":
            st.nCurValue = int(self.offset_x)
            st.nMin = 0
            st.nMax = int(self.sensor_w - self.width)
            st.nInc = int(self._inc)
            return 0
        if key == "OffsetY":
            st.nCurValue = int(self.offset_y)
            st.nMin = 0
            st.nMax = int(self.sensor_h - self.height)
            st.nInc = int(self._inc)
            return 0
        return 1

    def MV_CC_SetIntValue(self, key: str, value: int) -> int:
        v = int(value)
        if key in ("Width", "Height"):
            # 模拟：节点不可写（但值保持不变）
            return 1
        if key == "OffsetX":
            self.offset_x = v
            return 0
        if key == "OffsetY":
            self.offset_y = v
            return 0
        return 1


def test_configure_resolution_accepts_readonly_width_height_if_already_target() -> None:
    binding = _FakeBinding()
    cam = _FixedAoiReadOnlySizeCam()

    # 固定 AOI：宽高不可改，但我们仍希望能设置 offset。
    configure_resolution(
        binding=cast(Any, binding),
        cam=cam,
        width=1280,
        height=1080,
        offset_x=584,
        offset_y=484,
    )

    assert cam.width == 1280
    assert cam.height == 1080
    assert cam.offset_x == 584
    assert cam.offset_y == 484
