from __future__ import annotations

from tennis3d.trajectory.curve_stage import CurveStageConfig, apply_curve_stage


def _make_rec(*, t_abs: float, x: float, y: float, z: float) -> dict:
    # 说明：构造 run_localization_pipeline 的输出子集，覆盖 curve stage + episode。
    return {
        "capture_t_abs": float(t_abs),
        "created_at": 123.0,
        "balls": [
            {
                "ball_id": 0,
                "ball_3d_world": [float(x), float(y), float(z)],
                "quality": 1.0,
                # 提供部分诊断字段，确保观测过滤路径不会出错（这里不启用过滤）。
                "num_views": 3,
                "median_reproj_error_px": 1.0,
                "ball_3d_std_m": [0.01, 0.01, 0.01],
            }
        ],
    }


def _make_rec_two_balls(
    *,
    t_abs: float,
    # ball0
    x0: float,
    y0: float,
    z0: float,
    # ball1
    x1: float,
    y1: float,
    z1: float,
) -> dict:
    # 说明：构造“同一帧两个球”的输出，用于验证多 track + episode 锁定策略。
    return {
        "capture_t_abs": float(t_abs),
        "created_at": 123.0,
        "balls": [
            {
                "ball_id": 0,
                "ball_3d_world": [float(x0), float(y0), float(z0)],
                "quality": 1.0,
                "num_views": 3,
                "median_reproj_error_px": 1.0,
                "ball_3d_std_m": [0.01, 0.01, 0.01],
            },
            {
                "ball_id": 1,
                "ball_3d_world": [float(x1), float(y1), float(z1)],
                # 让 ball1 质量略低，确保贪心分配时 ball0 先处理。
                "quality": 0.8,
                "num_views": 3,
                "median_reproj_error_px": 1.0,
                "ball_3d_std_m": [0.01, 0.01, 0.01],
            },
        ],
    }


def test_curve_stage_episode_start_and_stationary_end() -> None:
    cfg = CurveStageConfig(
        enabled=True,
        primary="v2",  # 避免依赖 v3 的 predicted_land_time_abs 触发结束
        max_tracks=1,
        association_dist_m=10.0,
        max_missed_s=10.0,
        episode_enabled=True,
        # 起始：z 方向位移/速度门控 + y 轴重力一致性
        episode_buffer_s=1.0,
        episode_min_obs=5,
        episode_z_dir=-1,
        episode_min_abs_dz_m=0.25,
        episode_min_abs_vz_mps=1.0,
        episode_gravity_mps2=9.8,
        episode_gravity_tol_mps2=3.0,
        # 结束：低速持续一段时间
        episode_stationary_speed_mps=0.2,
        episode_end_if_stationary_s=0.25,
        # 只在 episode 活跃时喂给 curve，并在开始时回放
        feed_curve_only_when_episode_active=True,
        reset_predictor_on_episode_start=True,
        reset_predictor_on_episode_end=True,
    )

    t0 = 1000.0
    dt = 0.05
    g = 9.8
    y0 = 1.2
    v0 = 5.0
    z0 = 10.0
    vz = -3.0

    # 1) 准备阶段：基本不动（z 不变化，因此不会触发 episode_start）。
    recs = []
    for i in range(4):
        recs.append(_make_rec(t_abs=t0 + float(i) * dt, x=0.0, y=y0, z=z0))

    # 2) 击球飞行：y 为抛物线（重力一致），z 单调减小（朝机器人）。
    for j in range(8):
        tau = float(j) * dt
        y = float(y0 + v0 * tau - 0.5 * g * tau * tau)
        z = float(z0 + vz * tau)
        i = 4 + j
        recs.append(_make_rec(t_abs=t0 + float(i) * dt, x=0.2, y=y, z=z))

    # 3) 结束阶段：再次静止（达到 stationary 超时）。
    last_tau = float(7) * dt
    last_y = float(y0 + v0 * last_tau - 0.5 * g * last_tau * last_tau)
    last_z = float(z0 + vz * last_tau)
    for i in range(12, 21):
        recs.append(_make_rec(t_abs=t0 + float(i) * dt, x=0.2, y=last_y, z=last_z))

    outs = list(apply_curve_stage(recs, cfg))

    # 收集所有事件
    events: list[dict] = []
    for r in outs:
        c = r.get("curve")
        assert isinstance(c, dict)
        te = c.get("track_events")
        if isinstance(te, list):
            events.extend([e for e in te if isinstance(e, dict)])

    assert any(e.get("event") == "episode_start" for e in events)
    assert any((e.get("event") == "episode_end" and e.get("reason") == "stationary") for e in events)

    # 最后一帧应已结束 episode
    last = outs[-1]
    tu = last["curve"]["track_updates"]
    assert isinstance(tu, list) and len(tu) == 1
    ep = tu[0].get("episode")
    assert isinstance(ep, dict)
    assert bool(ep.get("active")) is False
    assert ep.get("end_reason") in {"stationary", "after_predicted_land", "track_drop_max_missed"}


def test_curve_stage_episode_start_physics_gate_gravity_and_z_dir() -> None:
    # 说明：
    # - y 为标准抛物线：y = y0 + v0*t - 0.5*g*t^2（ay=-g），满足“重力一致性”。
    # - z 单调减小，模拟“朝机器人飞来”（episode_z_dir=-1）。
    cfg = CurveStageConfig(
        enabled=True,
        primary="v2",
        max_tracks=1,
        association_dist_m=10.0,
        max_missed_s=10.0,
        episode_enabled=True,
        episode_buffer_s=2.0,
        episode_min_obs=5,
        episode_z_dir=-1,
        episode_min_abs_dz_m=0.3,
        episode_min_abs_vz_mps=1.0,
        episode_gravity_mps2=9.8,
        episode_gravity_tol_mps2=1.5,
        # 只在 episode 活跃时喂给 curve（这里不关心 curve 输出，只关心事件）
        feed_curve_only_when_episode_active=True,
    )

    t0 = 1000.0
    dt = 0.05
    g = 9.8
    y0 = 1.2
    v0 = 5.0
    z0 = 10.0
    vz = -3.0

    recs = []
    for i in range(7):
        t = float(i) * dt
        y = float(y0 + v0 * t - 0.5 * g * t * t)
        z = float(z0 + vz * t)
        recs.append(_make_rec(t_abs=t0 + t, x=0.2, y=y, z=z))

    outs = list(apply_curve_stage(recs, cfg))
    events: list[dict] = []
    for r in outs:
        te = r.get("curve", {}).get("track_events")
        if isinstance(te, list):
            events.extend([e for e in te if isinstance(e, dict)])

    starts = [e for e in events if e.get("event") == "episode_start"]
    assert starts, "physics gate should trigger episode_start"
    e0 = starts[0]
    assert e0.get("ay_mps2") is not None
    # ay 估计值应接近 -9.8（符号不强制，但数值应合理）。
    assert abs(abs(float(e0["ay_mps2"])) - 9.8) < 2.0


def test_curve_stage_episode_lock_single_track_prunes_others() -> None:
    cfg = CurveStageConfig(
        enabled=True,
        primary="v2",
        max_tracks=5,
        association_dist_m=10.0,
        max_missed_s=10.0,
        episode_enabled=True,
        episode_lock_single_track=True,
        episode_buffer_s=2.0,
        episode_min_obs=5,
        episode_z_dir=-1,
        episode_min_abs_dz_m=0.25,
        episode_min_abs_vz_mps=1.0,
        episode_gravity_mps2=9.8,
        episode_gravity_tol_mps2=3.0,
        # 为了让 episode 更干净，这里只在 episode_active 时喂给 curve（但本测试不依赖 curve 输出）。
        feed_curve_only_when_episode_active=True,
        reset_predictor_on_episode_start=True,
        reset_predictor_on_episode_end=True,
    )

    t0 = 1000.0
    dt = 0.05
    g = 9.8
    y0 = 1.2
    v0 = 5.0
    z0 = 10.0
    vz = -3.0

    # ball1：固定在远处，确保会形成第二条 track，但不会触发 episode（z 不变化）。
    x1, y1, z1 = 5.0, 1.0, 30.0

    recs = []
    for i in range(10):
        tau = float(i) * dt
        y = float(y0 + v0 * tau - 0.5 * g * tau * tau)
        z = float(z0 + vz * tau)
        recs.append(
            _make_rec_two_balls(
                t_abs=t0 + tau,
                x0=0.2,
                y0=y,
                z0=z,
                x1=x1,
                y1=y1,
                z1=z1,
            )
        )

    outs = list(apply_curve_stage(recs, cfg))

    # 找到 episode_start 发生的那一帧，并验证此后只保留 1 条 track。
    locked_index = None
    for idx, r in enumerate(outs):
        c = r.get("curve")
        assert isinstance(c, dict)
        te = c.get("track_events")
        if isinstance(te, list) and any(isinstance(e, dict) and e.get("event") == "episode_start" for e in te):
            locked_index = idx
            break

    assert locked_index is not None, "episode_start should happen"

    for r in outs[locked_index:]:
        c = r.get("curve")
        assert isinstance(c, dict)
        nat = c.get("num_active_tracks")
        assert nat is not None
        assert int(nat) == 1
        tu = c.get("track_updates")
        assert isinstance(tu, list) and len(tu) == 1

        balls = r.get("balls")
        assert isinstance(balls, list) and len(balls) == 2
        # ball0 应该被赋予 curve_track_id；ball1 在锁定后应不再被分配/维护。
        b0 = balls[0]
        b1 = balls[1]
        assert isinstance(b0, dict) and isinstance(b1, dict)
        assert b0.get("curve_track_id") is not None
        assert b1.get("curve_track_id") is None
