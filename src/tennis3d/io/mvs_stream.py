"""在线相机流抽象。

职责：
- 以 `mvs.QuadCapture` 为基础，产出 pipeline 统一格式：(meta, images_by_camera)

说明：
- 相机同步/组包由 `mvs.open_quad_capture` 内部完成（buffer、group_timeout、max_pending 等）。
- 该模块只把“在线组包结果”转换为业务 pipeline 可消费的形式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from mvs import MvsBinding, QuadCapture

from tennis3d.pipeline import iter_mvs_image_groups


@dataclass(frozen=True)
class CameraStream:
    """在线相机流（按同步组输出）。"""

    cap: QuadCapture
    binding: MvsBinding

    def iter_groups(self, *, max_groups: int = 0, timeout_s: float = 0.5) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
        yield from iter_mvs_image_groups(
            cap=self.cap,
            binding=self.binding,
            max_groups=int(max_groups),
            timeout_s=float(timeout_s),
        )
