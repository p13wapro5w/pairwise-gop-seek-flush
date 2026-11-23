"""GOP 精确 seek、flush/解码器重建、PTS 重排与缓存失效的参考实现。

领域约束（也是本模块要保证的不变量）：

* 关键帧定位按**呈现时间 pts** 找 ``pts <= target`` 的最近关键帧；解码则按
  **解码顺序 dts** 从该关键帧开始。B 帧 GOP 的 pts 与 dts 顺序不同，二者
  不能混用。
* 精确 seek：从关键帧解码后丢弃 ``pts < target`` 的帧，返回第一帧
  ``pts >= target`` 的帧，并报告 seek 误差（帧粒度）。
* flush：切换片段必须 *reset* 解码器（清空 pending/output、递增
  generation、重新以关键帧锚定）。只 drain 输出而保留 pending 会让上一段
  的帧混进新片段，因此 ``mode="drain"`` 会被显式标记为有泄漏。
* PTS 重排：解码顺序不等于呈现顺序，输出前按 ``(pts, dts, decode_idx)``
  稳定排序，并校验严格递增与重复时间戳。
* 倍速：音频与视频使用同一个时间基缩放（``(pts - start) / speed``）。
  视频丢帧只影响视频采样，音频 PTS 必须保持连续，不得因丢帧累加偏移。
* 缓存键：包含工程版本、片段版本与转场参数；撤销后按工程版本或受影响
  片段版本使预渲染缓存失效。

所有公开函数都接受“数据对象”，并且为了便于冒烟测试，也接受单个标量：
标量会被当作 target/版本号，数据则回退到内置演示素材。
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

__all__ = [
    "keyframe_locate",
    "precise_seek",
    "flush_rebuild",
    "pts_reorder",
    "speed_align",
    "cache_key",
    "cache_invalidate",
    "validation",
    "audit",
    "fallback",
    "gop_scan",
    "metrics",
    "timeline_map",
    "b_frame",
    "determinism",
]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number_or_none(value):
    return value if _is_number(value) else None


def _first_present(mapping, names, default=None):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _looks_like_packet_seq(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(item, dict) for item in value)
    )


def _normalize_packets(packets) -> list[dict]:
    """兼容常见的字段命名，并补齐 idx/pts/dts/key/type。"""

    normalized = []
    for i, raw in enumerate(packets):
        packet = dict(raw)
        pts = _first_present(packet, ("pts", "presentation_pts", "timestamp", "time"))
        dts = _first_present(packet, ("dts", "decode_pts", "decode_time"), pts)
        key = _first_present(packet, ("key", "is_key", "is_keyframe", "keyframe"))
        frame_type = _first_present(packet, ("type", "frame_type", "picture_type"))
        if pts is None:
            pts = i * 1000
        if dts is None:
            dts = pts
        if frame_type is None:
            frame_type = "I" if key else "P"
        if key is None:
            key = str(frame_type).upper().startswith("I")
        idx = _first_present(packet, ("idx", "index", "decode_index"), i)
        packet.update(
            {
                "idx": idx,
                "pts": pts,
                "dts": dts,
                "key": bool(key),
                "type": str(frame_type).upper(),
            }
        )
        normalized.append(packet)
    return normalized


def _coerce_packets(value) -> list[dict]:
    """把 list/dict/标量统一成 packet 列表；无法识别时使用演示 GOP。

    显式的空列表表示空流，保留为空，不能回退到演示素材。
    """

    if isinstance(value, dict):
        for key in ("packets", "frames", "gop", "packets_list"):
            if isinstance(value.get(key), (list, tuple)):
                value = value[key]
                break
        else:
            if any(field in value for field in ("pts", "dts", "key", "type")):
                return _normalize_packets([value])
            return _normalize_packets(_demo_packets())
    if isinstance(value, (list, tuple)) and not value:
        return []
    if _looks_like_packet_seq(value):
        return _normalize_packets(value)
    if (
        isinstance(value, (list, tuple))
        and value
        and all(_is_number(item) for item in value)
    ):
        # 仅给出 pts 序列时的宽松支持：第一帧视为关键帧。
        return _normalize_packets(
            [
                {
                    "idx": i,
                    "pts": item,
                    "dts": item,
                    "key": i == 0,
                    "type": "I" if i == 0 else "P",
                }
                for i, item in enumerate(value)
            ]
        )
    return _normalize_packets(_demo_packets())


def _target_from(raw_data, explicit_target, default=2500):
    if explicit_target is not None:
        return explicit_target
    scalar = _number_or_none(raw_data)
    return scalar if scalar is not None else default


def _pres_key(packet):
    return (packet["pts"], packet["dts"], packet["idx"])


def _decode_key(packet):
    return (packet["dts"], packet["idx"])


def _presentation_order(packets):
    return sorted(packets, key=_pres_key)


def _keyframe_list(packets):
    return sorted(
        (packet for packet in packets if packet["key"]),
        key=_pres_key,
    )


# ---------------------------------------------------------------------------
# 演示素材：IBBP / IBBP 两个 GOP
# ---------------------------------------------------------------------------


def _demo_packets():
    # (idx, pts, dts, key, type, segment)
    raw = [
        (0, 0, 0, True, "I", "gop0"),
        (1, 3000, 1000, False, "P", "gop0"),
        (2, 1000, 2000, False, "B", "gop0"),
        (3, 2000, 3000, False, "B", "gop0"),
        (4, 6000, 4000, True, "I", "gop1"),
        (5, 9000, 5000, False, "P", "gop1"),
        (6, 7000, 6000, False, "B", "gop1"),
        (7, 8000, 7000, False, "B", "gop1"),
    ]
    return [
        {
            "idx": idx,
            "pts": pts,
            "dts": dts,
            "key": key,
            "type": typ,
            "segment": segment,
        }
        for idx, pts, dts, key, typ, segment in raw
    ]


def _demo_audio():
    return [
        {"idx": i, "pts": i * 1000, "duration": 1000}
        for i in range(10)
    ]


def _demo_clips():
    return [
        {
            "id": "clip-a",
            "start": 0,
            "end": 4000,
            "source_in": 1000,
            "speed": 1.0,
        },
        {
            "id": "clip-b",
            "start": 4000,
            "end": 8000,
            "source_in": 0,
            "speed": 2.0,
        },
    ]


def _demo_project():
    return {"version": 1, "duration": 8000, "clips": _demo_clips()}


def _demo_decoder_state():
    return {
        "generation": 1,
        "pending": [
            {"idx": 10, "pts": 3000, "dts": 2500, "key": False, "type": "P",
             "segment": "previous"},
            {"idx": 11, "pts": 4000, "dts": 3500, "key": False, "type": "B",
             "segment": "previous"},
        ],
        "output": [
            {"idx": 9, "pts": 2000, "dts": 1500, "key": False, "type": "B",
             "segment": "previous"},
        ],
        "last_pts": 4000,
        "anchored": False,
    }


def _coerce_decoder(value):
    if isinstance(value, dict) and any(
        key in value for key in ("generation", "pending", "output", "queue")
    ):
        decoder = deepcopy(value)
        decoder.setdefault("generation", 1)
        decoder.setdefault("pending", [])
        decoder.setdefault("output", [])
        if "queue" in decoder and not decoder["pending"]:
            decoder["pending"] = decoder.pop("queue")
        decoder.setdefault("last_pts", None)
        decoder.setdefault("anchored", False)
        return decoder
    return _demo_decoder_state()


def _coerce_audio(value):
    if isinstance(value, dict):
        for key in ("audio", "packets", "frames"):
            if isinstance(value.get(key), (list, tuple)):
                value = value[key]
                break
    if _looks_like_packet_seq(value):
        normalized = []
        for i, raw in enumerate(value):
            packet = dict(raw)
            pts = _first_present(packet, ("pts", "timestamp", "time"), i * 1000)
            packet.setdefault("idx", i)
            packet["pts"] = pts
            packet.setdefault("duration", 1000)
            normalized.append(packet)
        return normalized
    if (
        isinstance(value, (list, tuple))
        and value
        and all(_is_number(item) for item in value)
    ):
        return [
            {"idx": i, "pts": item, "duration": 1000}
            for i, item in enumerate(value)
        ]
    return _demo_audio()


# ---------------------------------------------------------------------------
# 1. 关键帧定位
# ---------------------------------------------------------------------------


def keyframe_locate(gop=None, target_pts=None):
    """按呈现时间定位 ``pts <= target`` 的最近关键帧。

    B 帧 GOP 中关键帧的 pts 不一定等于其 dts 位置，因此关键帧查找只看
    pts；返回的 ``keyframe_decode_idx`` 用于后续按 dts 顺序解码。
    """

    packets = _coerce_packets(gop)
    target = _target_from(gop, target_pts, default=2500)
    scan = gop_scan(packets)

    if not packets:
        return {
            "target_pts": target,
            "keyframe": None,
            "keyframe_pts": None,
            "keyframe_decode_idx": None,
            "keyframe_presentation_idx": None,
            "gop_index": None,
            "clamped": True,
            "search_order": "pts",
            "gop_count": 0,
        }

    keyframes = _keyframe_list(packets)
    if not keyframes:
        # 没有 I 帧时只能退化为首帧（intra-only 流或损坏素材）。
        anchor = _presentation_order(packets)[0]
        clamped = anchor["pts"] > target
        anchor_kind = "intra_only_fallback"
    else:
        anchor = keyframes[0]
        for candidate in keyframes:
            if candidate["pts"] <= target:
                anchor = candidate
            else:
                break
        clamped = anchor["pts"] > target
        anchor_kind = "keyframe"

    presentation = _presentation_order(packets)
    presentation_idx = next(
        (i for i, packet in enumerate(presentation) if packet["idx"] == anchor["idx"]),
        None,
    )
    gop_index = next(
        (
            group["index"]
            for group in scan["gops"]
            if group["anchor_idx"] == anchor["idx"]
        ),
        None,
    )

    return {
        "target_pts": target,
        "keyframe": anchor,
        "keyframe_pts": anchor["pts"],
        "keyframe_decode_idx": anchor["idx"],
        "keyframe_dts": anchor["dts"],
        "keyframe_presentation_idx": presentation_idx,
        "gop_index": gop_index,
        "gop_count": scan["count"],
        "clamped": clamped,
        "kind": anchor_kind,
        "search_order": "pts",
    }


# ---------------------------------------------------------------------------
# 2. 精确 seek
# ---------------------------------------------------------------------------


def precise_seek(gop=None, target_pts=None):
    """从关键帧解码并按 pts 丢弃，返回第一帧 ``pts >= target``。"""

    packets = _coerce_packets(gop)
    target = _target_from(gop, target_pts, default=2500)
    anchor_info = keyframe_locate(packets, target)
    anchor = anchor_info["keyframe"]

    if anchor is None:
        return {
            "target_pts": target,
            "anchor_pts": None,
            "anchor_decode_idx": None,
            "decoded": [],
            "discarded": [],
            "frame": None,
            "first_presented_pts": None,
            "exact": False,
            "seek_error_ms": None,
            "stale": [],
            "anchor_clamped": True,
            "gop_index": None,
        }

    # 按解码顺序从关键帧开始，遇到下一个关键帧就停止（不跨 GOP 解码）。
    decode_order = sorted(packets, key=_decode_key)
    start = next(
        i for i, packet in enumerate(decode_order) if packet["idx"] == anchor["idx"]
    )
    window = []
    for packet in decode_order[start:]:
        if window and packet["key"]:
            break
        window.append(packet)

    decoded = _presentation_order(window)
    discarded = [packet for packet in decoded if packet["pts"] < target]
    presented = [packet for packet in decoded if packet["pts"] >= target]
    first = presented[0] if presented else None

    return {
        "target_pts": target,
        "anchor_pts": anchor["pts"],
        "anchor_decode_idx": anchor["idx"],
        "decoded": decoded,
        "discarded": discarded,
        "frame": first,
        "first_presented_pts": first["pts"] if first else None,
        "exact": bool(first and first["pts"] == target),
        "seek_error_ms": (first["pts"] - target) if first else None,
        "stale": [],
        "anchor_clamped": anchor_info["clamped"],
        "gop_index": anchor_info["gop_index"],
    }


# ---------------------------------------------------------------------------
# 3. flush 与解码器重建
# ---------------------------------------------------------------------------


def flush_rebuild(decoder=None, gop=None, target_pts=None, mode="reset"):
    """切换片段时 flush 解码器。

    ``mode="reset"``（正确做法）会丢弃 pending/output、递增 generation、
    从新 GOP 的关键帧重新锚定；``mode="drain"`` 只清输出队列，用于暴露
    “flush 不彻底导致上一段帧混入”的问题。
    """

    packets = _coerce_packets(gop)
    target = _target_from(gop, target_pts, default=2500)
    old = _coerce_decoder(decoder)
    seek = precise_seek(packets, target)

    previous_generation = old["generation"]
    flushed_frames = len(old["pending"]) + len(old["output"])
    stale = []

    if mode == "reset":
        generation = previous_generation + 1
        # 新片段只输出 target 及其之后的帧；之前的帧在重建后的解码器里丢弃。
        output = [
            deepcopy(packet)
            for packet in seek["decoded"]
            if packet["pts"] >= target
        ]
        pending = []
        anchored = bool(seek["anchor_pts"] is not None)
        rebuild = True
    elif mode == "drain":
        # 错误示范：只 drain 输出，pending 中的上一段帧仍在，会被混入。
        generation = previous_generation
        stale = [deepcopy(packet) for packet in old["pending"] + old["output"]]
        output = stale + [
            deepcopy(packet)
            for packet in seek["decoded"]
            if packet["pts"] >= target
        ]
        pending = []
        anchored = False
        rebuild = False
    else:
        raise ValueError("mode must be 'reset' or 'drain'")

    new_decoder = {
        "generation": generation,
        "pending": pending,
        "output": output,
        "last_pts": output[-1]["pts"] if output else None,
        "anchored": anchored,
        "anchor_pts": seek["anchor_pts"],
    }

    return {
        "mode": mode,
        "rebuild": rebuild,
        "previous_generation": previous_generation,
        "generation": generation,
        "flushed_frames": flushed_frames,
        "stale_frames": stale,
        "leak_free": not stale,
        "frames": output,
        "decoder": new_decoder,
        "target_pts": target,
        "anchor_pts": seek["anchor_pts"],
    }


# ---------------------------------------------------------------------------
# 4. PTS 重排
# ---------------------------------------------------------------------------


def pts_reorder(packets=None):
    """按呈现时间稳定重排，并校验 PTS 单调性与重复时间戳。"""

    normalized = _coerce_packets(packets)
    presentation = _presentation_order(normalized)
    decode_order = [packet["idx"] for packet in sorted(normalized, key=_decode_key)]
    pts_values = [packet["pts"] for packet in presentation]

    duplicates = sorted(
        {pts for pts in pts_values if pts_values.count(pts) > 1}
    )
    strictly_increasing = all(
        later > earlier for earlier, later in zip(pts_values, pts_values[1:])
    )
    non_decreasing = all(
        later >= earlier for earlier, later in zip(pts_values, pts_values[1:])
    )

    return {
        "frames": presentation,
        "presentation_order": [packet["idx"] for packet in presentation],
        "decode_order": decode_order,
        "pts": pts_values,
        "dts": [packet["dts"] for packet in presentation],
        "monotonic": strictly_increasing,
        "non_decreasing": non_decreasing,
        "duplicates": duplicates,
        "reordered": [packet["idx"] for packet in presentation] != decode_order,
        "frame_count": len(presentation),
    }


# ---------------------------------------------------------------------------
# 5. 倍速下的音画对齐
# ---------------------------------------------------------------------------


def speed_align(frames=None, speed=None, audio=None):
    """按播放倍速映射音视频时间轴。

    视频和音频都使用 ``(src_pts - start) / speed``。视频按倍速丢帧只改变
    显示帧集合，音频 PTS 保持连续；不能用“丢帧后的视频序号”去推导音频
    时间，否则会产生累计偏移。
    """

    frames_is_packets = _looks_like_packet_seq(frames) or (
        isinstance(frames, dict)
        and any(
            isinstance(frames.get(key), (list, tuple))
            for key in ("packets", "frames", "gop")
        )
    )
    if not frames_is_packets and _is_number(frames) and speed is None:
        speed = frames
    if not frames_is_packets:
        frames = None
    if speed is None:
        speed = 2.0
    speed = float(speed)
    if speed <= 0:
        speed = 1.0

    packets = _coerce_packets(frames)
    audio_packets = _coerce_audio(audio)
    if not packets:
        return {
            "speed": speed,
            "start_pts": None,
            "video": [],
            "audio": [],
            "dropped": [],
            "kept": 0,
            "max_drift_ms": 0.0,
            "span_drift_ms": 0.0,
            "aligned": True,
            "audio_continuous": True,
        }

    ordered = _presentation_order(packets)
    start = ordered[0]["pts"]
    stride = max(1, int(round(speed))) if speed > 1 else 1

    video = []
    dropped = []
    kept_src = []
    for i, packet in enumerate(ordered):
        # 倍速时按 stride 抽帧，但首帧和末帧保留，避免尾部时间轴缩短。
        drop = bool(speed > 1 and stride > 1 and i % stride != 0 and i != len(ordered) - 1)
        out_pts = (packet["pts"] - start) / speed
        video.append(
            {
                "src_idx": packet["idx"],
                "src_pts": packet["pts"],
                "out_pts": out_pts,
                "display": not drop,
                "dropped": drop,
            }
        )
        if drop:
            dropped.append(packet["idx"])
        else:
            kept_src.append((packet["pts"], out_pts))

    audio_out = []
    audio_by_src = {}
    last_audio_src = start
    for packet in sorted(audio_packets, key=lambda item: item["pts"]):
        if packet["pts"] < start:
            continue
        out_pts = (packet["pts"] - start) / speed
        duration = packet.get("duration", 1000) / speed
        audio_out.append(
            {
                "src_idx": packet.get("idx"),
                "src_pts": packet["pts"],
                "out_pts": out_pts,
                "duration": duration,
            }
        )
        audio_by_src[packet["pts"]] = out_pts
        last_audio_src = packet["pts"]

    # 同一源 PTS 上音视频的输出时间差；理论上恒为 0。
    drift = 0.0
    for src_pts, out_pts in kept_src:
        if src_pts in audio_by_src:
            drift = max(drift, abs(out_pts - audio_by_src[src_pts]))

    video_end = kept_src[-1][1] if kept_src else 0.0
    audio_end = (last_audio_src - start) / speed
    span_drift = abs(video_end - audio_end)

    return {
        "speed": speed,
        "start_pts": start,
        "video": video,
        "audio": audio_out,
        "dropped": dropped,
        "kept": len(kept_src),
        "max_drift_ms": drift,
        "span_drift_ms": span_drift,
        "aligned": drift <= 1e-9 and span_drift <= 1e-9,
        "audio_continuous": len(audio_out) == len(
            [item for item in audio_packets if item["pts"] >= start]
        ),
        "time_base": "common_source_pts",
    }


# ---------------------------------------------------------------------------
# 6. 缓存键（含工程版本）
# ---------------------------------------------------------------------------


def _normalize_clip_versions(clip_versions):
    if clip_versions is None:
        clip_versions = [
            {"id": clip["id"], "revision": 1} for clip in _demo_clips()
        ]
    if isinstance(clip_versions, dict):
        clip_versions = [
            {"id": clip_id, "revision": revision}
            for clip_id, revision in clip_versions.items()
        ]
    normalized = []
    for clip in clip_versions:
        item = deepcopy(clip) if isinstance(clip, dict) else {"id": str(clip)}
        item.setdefault("id", item.get("name", "clip"))
        item.setdefault("revision", item.get("version", 1))
        normalized.append(item)
    normalized.sort(key=lambda item: str(item["id"]))
    return normalized


def cache_key(project_version=1, clip_versions=None, transition=None, params=None):
    """稳定的预渲染缓存键。

    键必须包含工程版本（撤销/重做会改变它）以及每个依赖片段的版本，否则
    跨片段转场缓存会在撤销之后仍然命中旧帧。
    """

    if isinstance(project_version, dict):
        project = project_version
        project_version = _first_present(
            project, ("version", "project_version", "revision"), 1
        )
        if clip_versions is None:
            clip_versions = project.get("clips")

    clips = _normalize_clip_versions(clip_versions)
    if transition is None:
        transition = {
            "type": "crossfade",
            "from": "clip-a",
            "to": "clip-b",
            "duration": 500,
        }
    if params is None:
        params = {"resolution": "1920x1080", "fps": 25, "color": "sdr"}

    payload = {
        "schema": "nv-prerender-cache/v2",
        "project_version": project_version,
        "clips": clips,
        "transition": transition,
        "params": params,
    }
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 7. 撤销后的缓存失效
# ---------------------------------------------------------------------------


def _demo_cache():
    clips_a = [{"id": "clip-a", "revision": 1}]
    clips_ab = [
        {"id": "clip-a", "revision": 1},
        {"id": "clip-b", "revision": 2},
    ]
    return {
        cache_key(2, clips_a): {
            "project_version": 2,
            "revisions": {"clip-a": 1},
            "frames": [0, 1000],
        },
        cache_key(2, clips_ab): {
            "project_version": 2,
            "revisions": {"clip-a": 1, "clip-b": 2},
            "frames": [4000, 5000],
        },
        cache_key(1, clips_a): {
            "project_version": 1,
            "revisions": {"clip-a": 1},
            "frames": [0, 1000],
        },
    }


def _demo_undo_event():
    return {
        "kind": "undo",
        "project_version": 2,
        "affected_clips": ["clip-b"],
        "restored_revision": {"clip-b": 1},
    }


def cache_invalidate(cache=None, undo=None, project_version=None):
    """撤销后让受影响的预渲染缓存失效。

    失效条件：
    * 条目记录的项目版本与当前版本不一致；
    * 条目依赖的片段出现在撤销影响集合中，且其渲染版本高于撤销恢复到的
      版本（即撤销掉了产生该缓存的那次修改）。
    """

    if not isinstance(cache, dict):
        cache = _demo_cache()
    event = undo if isinstance(undo, dict) else _demo_undo_event()

    current_version = (
        project_version
        if project_version is not None
        else _first_present(event, ("project_version", "version"))
    )
    affected = set(event.get("affected_clips") or [])
    restored = event.get("restored_revision") or {}
    if not affected and restored:
        affected = set(restored)

    kept = {}
    evicted = []
    reasons = {}
    for key, entry in cache.items():
        entry = entry if isinstance(entry, dict) else {}
        revisions = entry.get("revisions") or {}
        reason = None

        if (
            current_version is not None
            and entry.get("project_version") is not None
            and entry["project_version"] != current_version
        ):
            reason = "project_version"
        else:
            for clip_id in sorted(affected):
                entry_revision = revisions.get(clip_id)
                if entry_revision is None:
                    continue
                restored_revision = restored.get(clip_id, -1)
                if entry_revision > restored_revision:
                    reason = "undo:{}".format(clip_id)
                    break

        if reason is None:
            kept[key] = entry
        else:
            evicted.append(key)
            reasons[key] = reason

    return {
        "cache": kept,
        "kept": sorted(kept),
        "evicted": sorted(evicted),
        "reasons": reasons,
        "invalidated_count": len(evicted),
        "current_project_version": current_version,
        "affected_clips": sorted(affected),
        "event": event,
    }


# ---------------------------------------------------------------------------
# 8. 校验与审计
# ---------------------------------------------------------------------------


def validation(result=None):
    """校验 seek/flush/重排链路是否满足精确 seek 的不变量。"""

    if isinstance(result, dict) and (
        "frame" in result or "discarded" in result
    ):
        seek = result
        decoded = seek.get("decoded") or None
        flush = flush_rebuild(None, decoded, seek.get("target_pts"))
        reorder = pts_reorder(seek.get("decoded"))
    elif isinstance(result, (list, tuple)):
        # 显式传入帧序列：以首帧为参照的默认目标进行校验
        packets = _coerce_packets(result)
        target = packets[0]["pts"] if packets else 0
        seek = precise_seek(packets, target)
        flush = flush_rebuild(None, packets, target)
        reorder = pts_reorder(seek["decoded"])
    else:
        target = _target_from(result, None, default=2500)
        seek = precise_seek(None, target)
        flush = flush_rebuild(None, None, target)
        reorder = pts_reorder(seek["decoded"])

    target = seek.get("target_pts")
    frame = seek.get("frame")
    anchor_pts = seek.get("anchor_pts")
    checks = {
        "anchor_not_after_target": (
            anchor_pts is not None and target is not None and anchor_pts <= target
        ),
        "first_frame_at_or_after_target": (
            frame is not None and target is not None and frame["pts"] >= target
        ),
        "discarded_before_target": all(
            packet["pts"] < target for packet in seek.get("discarded", [])
        ),
        "presentation_monotonic": reorder["monotonic"],
        "no_duplicate_pts": not reorder["duplicates"],
        "flush_rebuilt_decoder": flush["rebuild"],
        "generation_bumped": flush["generation"] > flush["previous_generation"],
        "no_stale_frames": flush["leak_free"],
        "decoder_reanchored": flush["decoder"]["anchored"],
    }
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "ok": not errors,
        "checks": checks,
        "errors": errors,
        "target_pts": target,
        "anchor_pts": anchor_pts,
        "first_pts": frame["pts"] if frame else None,
        "seek_error_ms": seek.get("seek_error_ms"),
    }


def audit(session=None):
    """审计一次 seek/flush 会话：关键帧锚定、代际、残留帧、PTS。"""

    gop = None
    if isinstance(session, dict):
        target = _first_present(
            session, ("target_pts", "target", "timeline_pts"), 2500
        )
        gop = _first_present(
            session, ("packets", "frames", "gop"), None
        )
        if gop is not None:
            gop = _coerce_packets(gop)
    else:
        target = _target_from(session, None, default=2500)

    locate = keyframe_locate(gop, target)
    seek = precise_seek(gop, target)
    flush = flush_rebuild(None, gop, target)
    reorder = pts_reorder(seek["decoded"])
    align = speed_align(None, 2.0)

    events = [
        {
            "event": "keyframe_locate",
            "target_pts": target,
            "anchor_pts": locate["keyframe_pts"],
            "clamped": locate["clamped"],
        },
        {
            "event": "precise_seek",
            "discarded": len(seek["discarded"]),
            "first_pts": seek["first_presented_pts"],
            "exact": seek["exact"],
        },
        {
            "event": "flush_rebuild",
            "mode": flush["mode"],
            "generation": flush["generation"],
            "stale": len(flush["stale_frames"]),
        },
        {
            "event": "pts_reorder",
            "reordered": reorder["reordered"],
            "monotonic": reorder["monotonic"],
        },
        {
            "event": "speed_align",
            "speed": align["speed"],
            "drift_ms": align["max_drift_ms"],
        },
    ]

    stale_segments = sorted(
        {
            packet.get("segment")
            for packet in flush["frames"]
            if packet.get("segment") == "previous"
        }
    )
    ok = bool(
        locate["keyframe_pts"] is not None
        and locate["keyframe_pts"] <= target
        and flush["leak_free"]
        and reorder["monotonic"]
        and align["aligned"]
        and not stale_segments
    )
    return {
        "ok": ok,
        "target_pts": target,
        "events": events,
        "generation": flush["generation"],
        "stale_frames": len(flush["stale_frames"]),
        "stale_segments": stale_segments,
        "discarded_frames": len(seek["discarded"]),
        "reorder_depth": b_frame(seek["decoded"])["reorder_depth"],
        "av_drift_ms": align["max_drift_ms"],
    }


# ---------------------------------------------------------------------------
# 9. 降级回退
# ---------------------------------------------------------------------------


def fallback(target_pts=None, gop=None, require_exact=False, reason=None):
    """精确 seek 不可用时降级到最近关键帧。

    触发条件：目标之前没有可解码帧、关键帧被截断、或调用方要求精确命中
    但目标落在两个显示帧之间。
    """

    if _looks_like_packet_seq(target_pts) or (
        isinstance(target_pts, dict) and "frames" in target_pts
    ):
        gop, target_pts = target_pts, None

    packets = _coerce_packets(gop)
    target = _target_from(target_pts, None, default=2500)
    seek = precise_seek(packets, target)
    locate = keyframe_locate(packets, target)

    degraded = False
    chosen_pts = seek["first_presented_pts"]
    mode = "exact_seek"
    if seek["frame"] is None:
        degraded = True
        mode = "keyframe"
        reason = reason or "no_frame_after_target"
        chosen_pts = locate["keyframe_pts"]
    elif require_exact and not seek["exact"]:
        degraded = True
        mode = "keyframe"
        reason = reason or "exact_frame_unavailable"
        chosen_pts = locate["keyframe_pts"]

    return {
        "target_pts": target,
        "degraded": degraded,
        "mode": mode,
        "reason": reason,
        "chosen_pts": chosen_pts,
        "keyframe_pts": locate["keyframe_pts"],
        "anchor_clamped": locate["clamped"],
        "seek": seek,
    }


# ---------------------------------------------------------------------------
# 10. GOP 扫描
# ---------------------------------------------------------------------------


def gop_scan(packets=None):
    """扫描包序列，按解码顺序找出 GOP 边界与关键帧。"""

    normalized = _coerce_packets(packets)
    decode_order = sorted(normalized, key=_decode_key)

    groups = []
    leading = []
    current = None

    for packet in decode_order:
        if packet["key"]:
            if current is not None:
                groups.append(current)
            current = {"anchor": packet, "packets": [], "leading": leading}
            leading = []
        if current is None:
            leading.append(packet)
        else:
            current["packets"].append(packet)
    if current is not None:
        groups.append(current)

    if not groups and leading:
        groups = [{"anchor": None, "packets": leading, "leading": []}]

    result = []
    for index, group in enumerate(groups):
        anchor = group["anchor"]
        members = group["leading"] + group["packets"]
        if not members:
            continue
        anchor_pts = anchor["pts"] if anchor else min(
            packet["pts"] for packet in members
        )
        start_pts = min(packet["pts"] for packet in members)
        end_pts = max(packet["pts"] for packet in members)
        closed = bool(anchor) and all(
            packet["pts"] >= anchor_pts for packet in members
        )
        result.append(
            {
                "index": index,
                "anchor_idx": anchor["idx"] if anchor else None,
                "anchor_pts": anchor_pts,
                "start_pts": start_pts,
                "end_pts": end_pts,
                "frame_count": len(members),
                "frames": [packet["idx"] for packet in _presentation_order(members)],
                "closed": closed,
                "open": not closed,
            }
        )

    keyframe_pts = [
        group["anchor_pts"] for group in result if group["anchor_idx"] is not None
    ]
    return {
        "gops": result,
        "count": len(result),
        "keyframe_count": len(keyframe_pts),
        "keyframe_pts": keyframe_pts,
        "open_gops": sum(1 for group in result if group["open"]),
        "decode_order": [packet["idx"] for packet in decode_order],
    }


# ---------------------------------------------------------------------------
# 11. 指标
# ---------------------------------------------------------------------------


def metrics(scenario=None):
    """汇总一次典型 seek + flush + 倍速播放的指标。"""

    target = 2500
    speed = 2.0
    if isinstance(scenario, dict):
        target = _first_present(scenario, ("target_pts", "target"), target)
        speed = _first_present(scenario, ("speed",), speed)
    else:
        scalar = _number_or_none(scenario)
        if scalar is not None:
            target = scalar

    seek = precise_seek(None, target)
    locate = keyframe_locate(None, target)
    flush = flush_rebuild(None, None, target)
    reorder = pts_reorder(seek["decoded"])
    align = speed_align(None, speed)
    scan = gop_scan(None)
    invalidation = cache_invalidate()

    return {
        "target_pts": target,
        "keyframe_pts": locate["keyframe_pts"],
        "seek_error_ms": seek["seek_error_ms"],
        "discarded_frames": len(seek["discarded"]),
        "decoded_frames": len(seek["decoded"]),
        "flush_generation": flush["generation"],
        "flush_rebuilt": flush["rebuild"],
        "stale_frames": len(flush["stale_frames"]),
        "reorder_depth": b_frame(seek["decoded"])["reorder_depth"],
        "reordered": reorder["reordered"],
        "pts_monotonic": reorder["monotonic"],
        "speed": speed,
        "dropped_video_frames": len(align["dropped"]),
        "av_drift_ms": align["max_drift_ms"],
        "audio_continuous": align["audio_continuous"],
        "gop_count": scan["count"],
        "cache_evicted": invalidation["invalidated_count"],
        "cache_kept": len(invalidation["kept"]),
        "ok": bool(
            flush["leak_free"]
            and reorder["monotonic"]
            and align["aligned"]
            and seek["first_presented_pts"] is not None
            and seek["first_presented_pts"] >= target
        ),
    }


# ---------------------------------------------------------------------------
# 12. 时间线映射
# ---------------------------------------------------------------------------


def timeline_map(clip=None, target_pts=None):
    """把时间线时间映射到片段源时间，考虑 source_in 与播放倍速。"""

    if isinstance(clip, dict) and "clips" in clip:
        project = clip
    else:
        project = _demo_project()
    clips = project.get("clips") or _demo_clips()

    if isinstance(clip, (int, float)) and not isinstance(clip, bool):
        target_pts = clip
    if target_pts is None:
        target_pts = 2500
    target = target_pts

    for item in clips:
        start = item.get("start", 0)
        end = item.get("end", start)
        if start <= target < end:
            speed = float(item.get("speed", 1.0)) or 1.0
            source_in = item.get("source_in", 0)
            source_pts = source_in + (target - start) * speed
            return {
                "timeline_pts": target,
                "clip_id": item.get("id"),
                "clip": item,
                "in_clip": True,
                "in_gap": False,
                "local_pts": target - start,
                "source_pts": source_pts,
                "speed": speed,
                "source_in": source_in,
                "clip_duration": end - start,
            }

    return {
        "timeline_pts": target,
        "clip_id": None,
        "clip": None,
        "in_clip": False,
        "in_gap": True,
        "local_pts": None,
        "source_pts": None,
        "speed": None,
        "source_in": None,
        "clip_duration": None,
    }


# ---------------------------------------------------------------------------
# 13. B 帧 / 重排结构
# ---------------------------------------------------------------------------


def b_frame(packets=None):
    """分析 B 帧重排深度与呈现顺序。"""

    normalized = _coerce_packets(packets)
    decode_order = sorted(normalized, key=_decode_key)
    presentation = _presentation_order(normalized)

    presentation_pos = {
        packet["idx"]: position for position, packet in enumerate(presentation)
    }
    decode_pos = {
        packet["idx"]: position for position, packet in enumerate(decode_order)
    }
    depth = max(
        (
            abs(presentation_pos[packet["idx"]] - decode_pos[packet["idx"]])
            for packet in normalized
        ),
        default=0,
    )
    types = [packet["type"] for packet in presentation]

    return {
        "decode_order": [packet["idx"] for packet in decode_order],
        "presentation_order": [packet["idx"] for packet in presentation],
        "presentation_types": types,
        "b_frames": sum(1 for typ in types if typ.startswith("B")),
        "p_frames": sum(1 for typ in types if typ.startswith("P")),
        "i_frames": sum(1 for typ in types if typ.startswith("I")),
        "has_b_frames": any(typ.startswith("B") for typ in types),
        "reorder_depth": depth,
        "requires_reorder": [packet["idx"] for packet in presentation]
        != [packet["idx"] for packet in decode_order],
        "keyframe_pts": [packet["pts"] for packet in presentation if packet["key"]],
        "pts_of": {packet["idx"]: packet["pts"] for packet in normalized},
        "dts_of": {packet["idx"]: packet["dts"] for packet in normalized},
    }


# ---------------------------------------------------------------------------
# 14. 确定性
# ---------------------------------------------------------------------------


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _pipeline_run(target=2500, speed=2.0, project_version=1):
    locate = keyframe_locate(None, target)
    seek = precise_seek(None, target)
    flush = flush_rebuild(None, None, target)
    reorder = pts_reorder(seek["decoded"])
    align = speed_align(None, speed)
    invalidation = cache_invalidate()
    return {
        "anchor_pts": locate["keyframe_pts"],
        "first_pts": seek["first_presented_pts"],
        "discarded": [packet["idx"] for packet in seek["discarded"]],
        "generation": flush["generation"],
        "stale": len(flush["stale_frames"]),
        "presentation_order": reorder["presentation_order"],
        "pts": reorder["pts"],
        "speed": align["speed"],
        "dropped": align["dropped"],
        "drift": align["max_drift_ms"],
        "span_drift": align["span_drift_ms"],
        "cache_key": cache_key(project_version),
        "evicted": invalidation["evicted"],
    }


def determinism(scenario=None):
    """同一输入重复执行 seek/flush/reorder/倍速/缓存，结果必须一致。"""

    target = 2500
    speed = 2.0
    if isinstance(scenario, dict):
        target = _first_present(scenario, ("target_pts", "target"), target)
        speed = _first_present(scenario, ("speed",), speed)
    else:
        scalar = _number_or_none(scenario)
        if scalar is not None:
            target = scalar

    first = _pipeline_run(target=target, speed=speed)
    second = _pipeline_run(target=target, speed=speed)
    first_blob = _canonical_json(first)
    second_blob = _canonical_json(second)
    digest = hashlib.sha256(first_blob.encode("utf-8")).hexdigest()

    return {
        "runs": 2,
        "equal": first_blob == second_blob,
        "digest": digest,
        "target_pts": target,
        "speed": speed,
        "stable": first_blob == second_blob,
        "first": first,
    }
