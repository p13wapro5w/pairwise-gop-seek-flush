import pytest

import media.gop_seek_flush as mod


# --- 1. 关键帧定位：按 pts 找 <= target 的最近关键帧 -------------------------


def test_keyframe_locate_uses_pts_not_decode_index():
    # GOP: I(pts 0) P(pts 3000) B(pts 1000) B(pts 2000) | I(pts 6000) ...
    located = mod.keyframe_locate(None, 2500)
    assert located["keyframe_pts"] == 0
    assert located["keyframe_decode_idx"] == 0
    assert located["clamped"] is False

    located = mod.keyframe_locate(None, 6500)
    assert located["keyframe_pts"] == 6000
    # 第二个 GOP 的关键帧在解码顺序里的位置是 4，而不是呈现顺序里靠后的位置
    assert located["keyframe_decode_idx"] == 4


def test_keyframe_locate_target_before_first_keyframe_is_clamped():
    located = mod.keyframe_locate(None, -1)
    assert located["keyframe_pts"] == 0
    assert located["clamped"] is True


# --- 2. 精确 seek：解码后丢弃 pts < target 的帧 -------------------------------


def test_precise_seek_discards_early_frames_and_hits_first_visible():
    seek = mod.precise_seek(None, 2500)
    assert seek["anchor_pts"] == 0
    assert [p["pts"] for p in seek["discarded"]] == [0, 1000, 2000]
    assert seek["first_presented_pts"] == 3000
    assert all(p["pts"] < 2500 for p in seek["discarded"])
    assert seek["frame"]["pts"] >= 2500


def test_precise_seek_exact_hit_reports_zero_error():
    seek = mod.precise_seek(None, 3000)
    assert seek["exact"] is True
    assert seek["seek_error_ms"] == 0
    assert seek["first_presented_pts"] == 3000


def test_precise_seek_does_not_decode_past_next_keyframe():
    seek = mod.precise_seek(None, 2500)
    # 窗口必须只包含第一个 GOP，不能把下一个 GOP 的帧混进来
    assert {p["segment"] for p in seek["decoded"]} == {"gop0"}
    assert len(seek["decoded"]) == 4


# --- 3. flush 与解码器重建 ---------------------------------------------------


def test_flush_reset_increments_generation_and_drops_previous_frames():
    out = mod.flush_rebuild(None, None, 2500)
    assert out["mode"] == "reset"
    assert out["rebuild"] is True
    assert out["generation"] == out["previous_generation"] + 1
    assert out["leak_free"] is True
    assert out["stale_frames"] == []
    # 上一段的 pending/output 帧绝不能出现在新片段里
    assert all(f["segment"] != "previous" for f in out["frames"])
    assert out["decoder"]["anchored"] is True


def test_flush_drain_only_leaks_stale_frames():
    out = mod.flush_rebuild(None, None, 2500, mode="drain")
    assert out["rebuild"] is False
    assert out["leak_free"] is False
    assert out["stale_frames"]
    assert any(f["segment"] == "previous" for f in out["frames"])


def test_flush_rejects_unknown_mode():
    with pytest.raises(ValueError):
        mod.flush_rebuild(None, None, 2500, mode="fast")


# --- 4. PTS 重排 -------------------------------------------------------------


def test_pts_reorder_fixes_bbp_decode_order():
    result = mod.pts_reorder(None)
    assert result["decode_order"] != result["presentation_order"]
    pts = result["pts"]
    assert pts == sorted(pts)
    assert result["monotonic"] is True
    assert result["duplicates"] == []


def test_pts_reorder_flags_duplicate_timestamps():
    packets = [
        {"idx": 0, "pts": 0, "dts": 0, "key": True, "type": "I"},
        {"idx": 1, "pts": 100, "dts": 100, "type": "P"},
        {"idx": 2, "pts": 100, "dts": 200, "type": "P"},
    ]
    result = mod.pts_reorder(packets)
    assert result["duplicates"] == [100]
    assert result["monotonic"] is False


# --- 5. 倍速下的音画对齐 -----------------------------------------------------


def test_speed_align_keeps_audio_pts_continuous_when_video_drops_frames():
    out = mod.speed_align(None, 2.0)
    assert out["speed"] == 2.0
    assert out["dropped"], "2x 播放应抽掉部分视频帧"
    # 丢帧不能改变音频时间轴，源 PTS 相同的音视频映射必须一致
    assert out["max_drift_ms"] == 0.0
    assert out["span_drift_ms"] == 0.0
    assert out["aligned"] is True
    assert out["audio_continuous"] is True
    # 音频时间按 (pts - start) / speed 映射
    audio = {a["src_pts"]: a["out_pts"] for a in out["audio"]}
    assert audio[4000] == pytest.approx(2000.0)
    assert audio[8000] == pytest.approx(4000.0)


# --- 6. 缓存键含工程版本 -----------------------------------------------------


def test_cache_key_changes_with_project_version_and_clip_revision():
    base = mod.cache_key(1)
    assert base != mod.cache_key(2)
    assert base != mod.cache_key(1, [{"id": "clip-a", "revision": 2}])
    # 键与字典顺序无关，必须稳定
    assert mod.cache_key(3, {"b": 2, "a": 1}) == mod.cache_key(
        3, [{"id": "a", "revision": 1}, {"id": "b", "revision": 2}]
    )


# --- 7. 撤销后的缓存失效 -----------------------------------------------------


def test_cache_invalidate_evicts_entry_produced_by_undone_edit():
    before = mod.cache_key(2, [{"id": "clip-a", "revision": 1},
                               {"id": "clip-b", "revision": 2}])
    after = mod.cache_key(2, [{"id": "clip-a", "revision": 1},
                              {"id": "clip-b", "revision": 1}])
    cache = {
        before: {"project_version": 2, "revisions": {"clip-a": 1, "clip-b": 2}},
        after: {"project_version": 2, "revisions": {"clip-a": 1, "clip-b": 1}},
    }
    event = {
        "kind": "undo",
        "project_version": 2,
        "affected_clips": ["clip-b"],
        "restored_revision": {"clip-b": 1},
    }
    out = mod.cache_invalidate(cache, event)
    assert out["evicted"] == [before]
    assert out["kept"] == [after]
    assert out["reasons"][before] == "undo:clip-b"


def test_cache_invalidate_evicts_stale_project_version():
    old = mod.cache_key(1)
    current = mod.cache_key(2)
    cache = {
        old: {"project_version": 1, "revisions": {}},
        current: {"project_version": 2, "revisions": {}},
    }
    out = mod.cache_invalidate(cache, {"kind": "undo", "project_version": 2})
    assert out["evicted"] == [old]
    assert out["kept"] == [current]
    assert out["reasons"][old] == "project_version"


# --- 端到端不变量 ------------------------------------------------------------


def test_validation_passes_on_full_pipeline():
    result = mod.validation(None)
    assert result["ok"] is True, result["errors"]


def test_audit_reports_no_stale_frames_and_no_av_drift():
    result = mod.audit(None)
    assert result["stale_frames"] == 0
    assert result["stale_segments"] == []
    assert result["av_drift_ms"] == 0.0
    assert result["ok"] is True


def test_gop_scan_finds_two_gops():
    scan = mod.gop_scan(None)
    assert scan["count"] == 2
    assert scan["keyframe_pts"] == [0, 6000]
    assert scan["open_gops"] == 0


def test_determinism_is_stable_across_runs():
    result = mod.determinism(None)
    assert result["equal"] is True
    assert result["first"]["cache_key"] == mod.cache_key(1)


def test_timeline_map_applies_speed_and_source_in():
    # clip-b: 时间线 4000-8000，2x，source_in 0
    mapped = mod.timeline_map(None, 5000)
    assert mapped["clip_id"] == "clip-b"
    assert mapped["source_pts"] == pytest.approx(2000.0)
    assert mapped["speed"] == 2.0
    # 间隙
    gap = mod.timeline_map(None, 4000 + 8000)
    assert gap["in_gap"] is True


def test_b_frame_detects_reorder_requirement():
    result = mod.b_frame(None)
    assert result["has_b_frames"] is True
    assert result["requires_reorder"] is True
    assert result["reorder_depth"] == 2


def test_fallback_degrades_when_exact_window_missing():
    assert mod.fallback(3000)["degraded"] is False
    assert mod.fallback(3000)["mode"] == "exact_seek"
    exact_fallback = mod.fallback(2500, require_exact=True)
    assert exact_fallback["mode"] == "keyframe"
    assert exact_fallback["degraded"] is True
    assert exact_fallback["chosen_pts"] == 0
    # 超出末尾没有可呈现帧时回退到关键帧
    assert mod.fallback(20000)["mode"] == "keyframe"


def test_empty_stream_is_not_replaced_by_demo_data():
    assert mod.gop_scan([])["count"] == 0
    assert mod.precise_seek([])["frame"] is None
    assert mod.pts_reorder([])["frame_count"] == 0
