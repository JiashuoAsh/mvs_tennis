"""单测：在线滑窗时间映射（方案B）。

覆盖目标：
- warmup 后能拟合出每台相机的映射（a,b）
- 在滑动窗口内能保持较小残差
- 当时间关系发生缓慢变化时（模拟温漂/漂移），能通过滑窗更新跟上变化

说明：
- 该测试不依赖真实相机，只用合成数据模拟 (dev_timestamp, host_timestamp_ms)。
"""

from __future__ import annotations

import unittest

from mvs import OnlineDevToHostMapper


class TestOnlineDevToHostMapper(unittest.TestCase):
    def test_online_mapper_warmup_and_update(self) -> None:
        mapper = OnlineDevToHostMapper(
            warmup_groups=5,
            window_groups=40,
            update_every_groups=1,
            min_points=10,
            hard_outlier_ms=50.0,
        )

        a = 1.0e-5
        b0 = 1_700_000_000_000.0

        serials = ["CAM_A", "CAM_B"]
        dev = {s: 10_000_000_000 for s in serials}

        # 前半段：稳定关系；后半段：给 CAM_A 增加一个缓慢变化的偏移（模拟漂移）。
        for gi in range(80):
            for si, s in enumerate(serials):
                dev[s] += 100_000 + 10 * si

                # 确保无随机性：使用周期性小扰动（<=0.2ms）。
                noise_ms = ((gi % 5) - 2) * 0.1

                drift_ms = 0.0
                if s == "CAM_A" and gi >= 40:
                    # 40 组之后开始缓慢漂移，总共约 4ms。
                    drift_ms = (gi - 40) * 0.1

                host_ms = a * float(dev[s]) + b0 + drift_ms + noise_ms

                mapper.observe_pair(serial=s, dev_ts=int(dev[s]), host_ms=int(round(host_ms)))

            mapper.on_group_end()

        # warmup+更新后，应已具备两台相机的映射。
        self.assertGreaterEqual(mapper.ready_count(serials), 2)

        worst_p95 = mapper.worst_p95_ms(serials)
        self.assertIsNotNone(worst_p95)
        if worst_p95 is not None:
            # 合成噪声非常小，p95 不应太大（给一点宽容，避免不同 Python 浮点/round 细节影响）。
            self.assertLess(worst_p95, 10.0)

        # 检查 CAM_A 在漂移后期的预测：应能跟上最新窗口内的偏移。
        m = mapper.get_mapping("CAM_A")
        self.assertIsNotNone(m)
        if m is not None:
            pred = m.map_dev_to_host_ms(int(dev["CAM_A"]))
            true = a * float(dev["CAM_A"]) + b0 + (79 - 40) * 0.1
            self.assertLess(abs(pred - true), 8.0)


if __name__ == "__main__":
    unittest.main()
