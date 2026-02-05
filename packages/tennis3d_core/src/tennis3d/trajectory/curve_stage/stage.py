from __future__ import annotations

import math
from typing import Any

from tennis3d.trajectory.curve_stage.config import CurveStageConfig
from tennis3d.trajectory.curve_stage.curve_imports import _CurveImports, _ensure_curve_imports
from tennis3d.trajectory.curve_stage.episode import (
    _episode_maybe_release_lock,
    _episode_should_feed_curve,
    _episode_stationary_motion_update,
    _episode_trim_recent,
    _episode_try_end,
    _episode_try_start,
)
from tennis3d.trajectory.curve_stage.models import _RecentObs, _Track
from tennis3d.trajectory.curve_stage.obs_extract import _choose_t_abs, _extract_meas_list, _obs_conf
from tennis3d.trajectory.curve_stage.output import _track_snapshot
from tennis3d.trajectory.curve_stage.tracking import _dist3, _predict_track_pos


class CurveStage:
    """对 out_rec 流做曲线拟合增强。"""

    def __init__(self, cfg: CurveStageConfig) -> None:
        self._cfg = cfg
        self._next_track_id = 1
        self._tracks: list[_Track] = []

        # episode 锁定：当某条 track 进入 episode 后，可选地丢弃其它 track，只保留该条。
        self._episode_locked_track_id: int | None = None

        # 延迟 import：避免在未启用时引入 curve 相关依赖与启动开销。
        self._imports: _CurveImports | None = None

    def reset(self) -> None:
        """清空所有 track 状态。"""

        self._tracks.clear()
        self._next_track_id = 1
        self._episode_locked_track_id = None

    def _ensure_imports(self) -> _CurveImports:
        if self._imports is None:
            self._imports = _ensure_curve_imports(self._cfg)
        return self._imports

    def _new_v3(self, imps: _CurveImports) -> Any | None:
        if imps.CurvePredictorV3 is None:
            return None
        try:
            return imps.CurvePredictorV3()  # type: ignore[operator]
        except Exception:
            return None

    def _new_v2(self, imps: _CurveImports) -> Any | None:
        if imps.CurveV2 is None:
            return None
        try:
            return imps.CurveV2()  # type: ignore[operator]
        except Exception:
            return None

    def _new_v3_legacy(self, imps: _CurveImports) -> Any | None:
        if imps.CurveV3Legacy is None:
            return None
        try:
            return imps.CurveV3Legacy()  # type: ignore[operator]
        except Exception:
            return None

    def _reset_track_predictors(self, tr: _Track, imps: _CurveImports) -> None:
        """重置某条 track 的拟合器状态。"""

        if self._cfg.primary == "v3" and imps.CurvePredictorV3 is not None:
            tr.v3 = self._new_v3(imps)
        if (self._cfg.primary == "v2" or self._cfg.compare_v2) and imps.CurveV2 is not None:
            tr.v2 = self._new_v2(imps)
        if (self._cfg.primary == "v3_legacy" or self._cfg.compare_v3_legacy) and imps.CurveV3Legacy is not None:
            tr.v3_legacy = self._new_v3_legacy(imps)

        tr.predicted_land_time_abs = None
        tr.predicted_second_land_time_abs = None
        setattr(tr, "_last_v2_points", None)
        setattr(tr, "_last_v3_legacy_points", None)

    def _maybe_update_predicted_land_time_abs(self, tr: _Track) -> None:
        predictor = tr.v3 or tr.v3_legacy or tr.v2
        if predictor is None:
            return

        # 每次更新都显式覆盖，避免在分段/后验变化后残留旧值。
        tr.predicted_land_time_abs = None
        tr.predicted_second_land_time_abs = None

        time_base_abs = getattr(predictor, "time_base_abs", None)
        if not isinstance(time_base_abs, (int, float)):
            time_base_abs = getattr(predictor, "time_base", None)
        if not isinstance(time_base_abs, (int, float)):
            return

        # first land：v3 走 predicted_land_time_rel；v2/legacy 走 land_point[0][-1]
        t1_rel_f: float | None = None
        try:
            fn1 = getattr(predictor, "predicted_land_time_rel", None)
            if callable(fn1):
                t1 = fn1()
                if isinstance(t1, (int, float)):
                    t1_rel_f = float(t1)
        except Exception:
            t1_rel_f = None

        if t1_rel_f is None:
            lp0 = getattr(predictor, "land_point", None)
            try:
                if isinstance(lp0, list) and lp0 and lp0[0] is not None:
                    t1_rel_f = float(lp0[0][-1])
            except Exception:
                t1_rel_f = None

        if t1_rel_f is not None:
            try:
                tr.predicted_land_time_abs = float(time_base_abs + float(t1_rel_f))
            except Exception:
                tr.predicted_land_time_abs = None

        # second land：要求所有 curve 都提供 predicted_second_land_time_rel 接口。
        t2_rel_f: float | None = None
        try:
            fn2 = getattr(predictor, "predicted_second_land_time_rel", None)
            if callable(fn2):
                t2 = fn2()
                if isinstance(t2, (int, float)):
                    t2_rel_f = float(t2)
        except Exception:
            t2_rel_f = None

        if t2_rel_f is None:
            return
        try:
            tr.predicted_second_land_time_abs = float(time_base_abs + float(t2_rel_f))
        except Exception:
            tr.predicted_second_land_time_abs = None

    def _feed_track_predictors(self, tr: _Track, imps: _CurveImports, o: _RecentObs) -> None:
        # v3：新 API（落点/走廊）。
        if tr.v3 is not None and imps.BallObservation is not None:
            try:
                obs = imps.BallObservation(  # type: ignore[misc]
                    x=float(o.x),
                    y=float(o.y),
                    z=float(o.z),
                    t=float(o.t_abs),
                    conf=o.conf,
                )
                tr.v3.add_observation(obs)
            except Exception:
                pass

        # v2/v3_legacy：legacy API（接球点候选），记录“最后一次输出”。
        if tr.v2 is not None:
            try:
                pts = tr.v2.add_frame(
                    [float(o.x), float(o.y), float(o.z), float(o.t_abs)],
                    is_bot_fire=int(self._cfg.is_bot_fire),
                )
            except Exception:
                pts = None
            setattr(tr, "_last_v2_points", pts)

        if tr.v3_legacy is not None:
            try:
                pts = tr.v3_legacy.add_frame(
                    [float(o.x), float(o.y), float(o.z), float(o.t_abs)],
                    is_bot_fire=int(self._cfg.is_bot_fire),
                )
            except Exception:
                pts = None
            setattr(tr, "_last_v3_legacy_points", pts)

    def _transform_y(self, y: float) -> float:
        """对输入到 curve 的 y 做可选变换。"""

        y2 = float(y) - float(self._cfg.y_offset_m)
        if bool(self._cfg.y_negate):
            y2 = -float(y2)
        return float(y2)

    def _speed_mps(
        self,
        *,
        prev_pos: tuple[float, float, float] | None,
        prev_t: float | None,
        pos: tuple[float, float, float],
        t: float,
    ) -> float | None:
        if prev_pos is None or prev_t is None:
            return None
        dt = float(t - float(prev_t))
        if dt < float(self._cfg.min_dt_s):
            return None
        d = _dist3(pos, prev_pos)
        return float(d / dt) if dt > 0 else None

    def _episode_lock_to_track(
        self,
        *,
        keep_track_id: int,
        now_t_abs: float,
        events: list[dict[str, Any]],
        ended_tracks: list[dict[str, Any]],
    ) -> None:
        """进入 episode 后可选地把系统锁定到单条 track。"""

        if not bool(self._cfg.episode_enabled):
            return
        if not bool(self._cfg.episode_lock_single_track):
            return
        if self._episode_locked_track_id is not None:
            return

        keep: _Track | None = None
        dropped: list[_Track] = []
        for tr in self._tracks:
            if int(tr.track_id) == int(keep_track_id):
                keep = tr
            else:
                dropped.append(tr)
        if keep is None:
            return

        for tr in dropped:
            if bool(tr.episode_active):
                tr.episode_active = False
                tr.episode_end_t_abs = float(tr.last_t_abs) if tr.last_t_abs is not None else float(now_t_abs)
                tr.episode_end_reason = "track_drop_episode_lock"
                events.append(
                    {
                        "event": "episode_end",
                        "track_id": int(tr.track_id),
                        "episode_id": int(tr.episode_id),
                        "t_abs": float(tr.episode_end_t_abs),
                        "reason": str(tr.episode_end_reason),
                    }
                )
            events.append(
                {
                    "event": "track_drop",
                    "track_id": int(tr.track_id),
                    "t_abs": float(now_t_abs),
                    "reason": "episode_lock_single_track",
                }
            )
            ended_tracks.append(_track_snapshot(tr, self._cfg))

        self._tracks = [keep]
        self._episode_locked_track_id = int(keep.track_id)

    def process_record(self, out_rec: dict[str, Any]) -> dict[str, Any]:
        """处理单条定位输出记录，返回增强后的记录。"""

        if not bool(self._cfg.enabled):
            return out_rec

        imps = self._ensure_imports()
        t_abs, t_source = _choose_t_abs(out_rec)

        track_events: list[dict[str, Any]] = []
        ended_tracks: list[dict[str, Any]] = []

        # 先清理长期未更新的 track。
        if t_abs > 0:
            alive: list[_Track] = []
            for tr in self._tracks:
                if tr.last_t_abs is None:
                    alive.append(tr)
                    continue
                if float(t_abs - tr.last_t_abs) <= float(self._cfg.max_missed_s):
                    alive.append(tr)
                    continue

                if bool(self._cfg.episode_enabled):
                    if bool(tr.episode_active):
                        tr.episode_active = False
                        tr.episode_end_t_abs = float(tr.last_t_abs)
                        tr.episode_end_reason = "track_drop_max_missed"
                        track_events.append(
                            {
                                "event": "episode_end",
                                "track_id": int(tr.track_id),
                                "episode_id": int(tr.episode_id),
                                "t_abs": float(tr.episode_end_t_abs),
                                "reason": str(tr.episode_end_reason),
                            }
                        )
                    track_events.append(
                        {
                            "event": "track_drop",
                            "track_id": int(tr.track_id),
                            "t_abs": float(t_abs),
                            "reason": "max_missed_s",
                        }
                    )
                    ended_tracks.append(_track_snapshot(tr, self._cfg))

                if self._episode_locked_track_id is not None and int(tr.track_id) == int(self._episode_locked_track_id):
                    self._episode_locked_track_id = None
            self._tracks = alive

        meas = _extract_meas_list(out_rec, self._cfg)
        assignments: list[dict[str, Any]] = []
        updated_tracks: list[_Track] = []
        used_track_ids: set[int] = set()

        for m in meas:
            y_in = self._transform_y(float(m.y))
            pos = (float(m.x), float(y_in), float(m.z))

            best_tr: _Track | None = None
            best_d = float("inf")
            for tr in self._tracks:
                if tr.track_id in used_track_ids:
                    continue
                if tr.last_pos is None or tr.last_t_abs is None:
                    continue
                dt = float(t_abs - float(tr.last_t_abs))
                if dt < float(self._cfg.min_dt_s):
                    continue
                pred = _predict_track_pos(tr, t_abs=float(t_abs), min_dt_s=float(self._cfg.min_dt_s))
                if pred is None:
                    continue
                d = _dist3(pos, pred)
                if d <= float(self._cfg.association_dist_m) and d < best_d:
                    best_d = d
                    best_tr = tr

            if best_tr is None:
                if self._episode_locked_track_id is not None:
                    continue
                if len(self._tracks) >= int(self._cfg.max_tracks):
                    continue

                tr = _Track(track_id=int(self._next_track_id), created_t_abs=float(t_abs))
                self._next_track_id += 1

                if self._cfg.primary == "v3" and imps.CurvePredictorV3 is not None:
                    tr.v3 = self._new_v3(imps)

                if (self._cfg.primary == "v2" or self._cfg.compare_v2) and imps.CurveV2 is not None:
                    tr.v2 = self._new_v2(imps)

                if (self._cfg.primary == "v3_legacy" or self._cfg.compare_v3_legacy) and imps.CurveV3Legacy is not None:
                    tr.v3_legacy = self._new_v3_legacy(imps)

                self._tracks.append(tr)
                best_tr = tr

            if best_tr.last_t_abs is not None:
                dt = float(t_abs - float(best_tr.last_t_abs))
                if dt < float(self._cfg.min_dt_s):
                    continue

            conf = _obs_conf(self._cfg, m.quality)
            obs = _RecentObs(t_abs=float(t_abs), x=float(m.x), y=float(y_in), z=float(m.z), conf=conf)

            sp = self._speed_mps(prev_pos=best_tr.last_pos, prev_t=best_tr.last_t_abs, pos=pos, t=float(t_abs))
            if sp is not None and math.isfinite(float(sp)):
                best_tr.last_speed_mps = float(sp)
                if best_tr.speed_ewma_mps is None:
                    best_tr.speed_ewma_mps = float(sp)
                else:
                    best_tr.speed_ewma_mps = 0.8 * float(best_tr.speed_ewma_mps) + 0.2 * float(sp)

                _episode_stationary_motion_update(self._cfg, best_tr, speed_mps=float(sp), t_abs=float(t_abs))

            best_tr.recent.append(obs)
            _episode_trim_recent(self._cfg, best_tr, now_t_abs=float(t_abs))

            was_episode_active = bool(best_tr.episode_active)
            _episode_try_start(self._cfg, best_tr, events=track_events)

            if (not was_episode_active) and bool(best_tr.episode_active):
                # episode_start 触发后，根据配置做 predictor reset + 回放 recent 窗口。
                if bool(self._cfg.reset_predictor_on_episode_start):
                    self._reset_track_predictors(best_tr, imps)
                    if bool(self._cfg.feed_curve_only_when_episode_active):
                        k = int(self._cfg.episode_min_obs)
                        window = list(best_tr.recent)[-k:]
                        for o in window:
                            self._feed_track_predictors(best_tr, imps, o)

                self._episode_lock_to_track(
                    keep_track_id=int(best_tr.track_id),
                    now_t_abs=float(t_abs),
                    events=track_events,
                    ended_tracks=ended_tracks,
                )

            if _episode_should_feed_curve(self._cfg, best_tr):
                self._feed_track_predictors(best_tr, imps, obs)
                self._maybe_update_predicted_land_time_abs(best_tr)

            ended = _episode_try_end(self._cfg, best_tr, now_t_abs=float(t_abs), events=track_events)
            if ended and bool(self._cfg.reset_predictor_on_episode_end):
                self._reset_track_predictors(best_tr, imps)

            if bool(self._cfg.episode_enabled) and bool(self._cfg.episode_lock_single_track):
                self._episode_locked_track_id = _episode_maybe_release_lock(self._episode_locked_track_id, best_tr)

            best_tr.prev_t_abs = best_tr.last_t_abs
            best_tr.prev_pos = best_tr.last_pos
            best_tr.last_t_abs = float(t_abs)
            best_tr.last_pos = pos
            best_tr.n_obs += 1

            used_track_ids.add(int(best_tr.track_id))
            updated_tracks.append(best_tr)
            assignments.append(
                {
                    "ball_id": int(m.ball_id),
                    "track_id": int(best_tr.track_id),
                    "dist_m": float(best_d) if best_d < float("inf") else None,
                }
            )

        # 若发生了 episode 单 track 锁定，确保输出与回写只包含当前仍存活的 track。
        active_ids = {int(tr.track_id) for tr in self._tracks}
        if active_ids:
            updated_tracks = [tr for tr in updated_tracks if int(tr.track_id) in active_ids]
            assignments = [a for a in assignments if int(a.get("track_id", -1)) in active_ids]

        # 把 track_id 回写到 balls（便于下游按球筛选/回放）。
        balls = out_rec.get("balls")
        if isinstance(balls, list):
            bid_to_tid = {int(a["ball_id"]): int(a["track_id"]) for a in assignments}
            for b in balls:
                if isinstance(b, dict):
                    bid = int(b.get("ball_id", -1))
                    if bid in bid_to_tid:
                        b["curve_track_id"] = int(bid_to_tid[bid])


        out_rec["curve"] = {
            "primary": str(self._cfg.primary),
            "t_abs": float(t_abs),
            "t_source": str(t_source),
            "assignments": assignments,
            "track_updates": [_track_snapshot(tr, self._cfg) for tr in updated_tracks],
            "num_active_tracks": int(len(self._tracks)),
        }

        if bool(self._cfg.episode_enabled):
            out_rec["curve"]["track_events"] = track_events
            out_rec["curve"]["ended_tracks"] = ended_tracks

        return out_rec
