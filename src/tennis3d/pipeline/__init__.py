"""Online/offline shared pipeline for tennis ball 3D localization.

This package hosts reusable pipeline building blocks:
- sources: produce per-group images from online MVS stream or offline captures.
- core: run detection + multi-view localization and emit JSON-serializable records.

The goal is to keep `tennis3d.apps.*` thin (CLI only).
"""

from .core import run_localization_pipeline
from .sources import iter_capture_image_groups, iter_mvs_image_groups

__all__ = [
    "iter_capture_image_groups",
    "iter_mvs_image_groups",
    "run_localization_pipeline",
]
