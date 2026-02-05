"""旧在线包已移除。

说明：
- 在线应用已拆分到独立顶层包 `tennis3d_online`。
- 请使用：`python -m tennis3d_online --help` 或 console script `tennis3d-online`。
"""

raise ImportError(
	"`tennis3d.apps.online` 已拆分为独立包 `tennis3d_online`；"
	"请改用 `python -m tennis3d_online` 或 `tennis3d-online`。"
)
