"""Offline flood-video understanding: YOLO+ByteTrack + FloodDetector + VLM keyframes.

Lightweight alternative to the full pipeline for a recorded clip. Loads only
the shared detector/tracker, the cheap flood heuristic and the VLM analyzer
(no pose / face / ReID / ...). Every frame is read so the tracker keeps
continuity; the VLM only runs on a handful of keyframes picked cheaply:

  * the first frame after 0.5 s of video,
  * then every --interval seconds of video time (stretched to
    duration / --max-keyframes for long videos),
  * immediately on FloodDetector event onset (first event, or first after
    10 s without one),
  * on a large scene change (64x36 gray thumbnail mean abs diff > 18), at most
    once per interval/2 and only from budget not reserved for the remaining
    periodic keyframes,
  * never more than --max-keyframes in total.

Output: data/flood_reports/<video_stem>_<YYYYmmdd_HHMMSS>/
  keyframe_XX_t{t}.jpg   image the VLM saw (numbered actor boxes)
  report.json            video meta, keyframes, per-actor tracks, summary
  summary.txt            human-readable version

Usage:
  python run_flood_video.py VIDEO [--profile flood] [--interval 2.0]
         [--max-keyframes 8] [--bitchat] [--show] [--out data/flood_reports]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

_ALERT_ORDER = {"unknown": -1, "none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
_SCENE_CHANGE_DIFF = 18.0          # mean abs diff on a 64x36 gray thumbnail
_SCENE_CHANGE_MIN_GAP_S = 0.5      # don't fire scene-change keyframes back to back
_FIRST_KEYFRAME_S = 0.5
_FLOOD_EVERY = 5                   # FloodDetector cadence (frames)
_FLOOD_ONSET_GAP_S = 10.0          # a FLOOD event counts as onset after this much quiet

_COCO_SHORT = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
               8: "boat", 16: "dog", 17: "horse", 19: "cow", 24: "backpack", 26: "handbag",
               28: "suitcase", 39: "bottle"}


def _alert_rank(level: Any) -> int:
    return _ALERT_ORDER.get(str(level or "unknown").lower(), -1)


def _thumb(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    return g.astype(np.int16)


def _actor_line(a: dict) -> str:
    parts = [f"#{a.get('id', '?')} {a.get('kind', '')}:".strip()]
    for k in ("state", "action"):
        v = (a.get(k) or "").strip() if isinstance(a.get(k), str) else a.get(k)
        if v:
            parts.append(str(v))
    if a.get("risk") not in (None, ""):
        parts.append(f"(risk {a.get('risk')})")
    return " ".join(parts)


def _jsonable(obj: Any) -> Any:
    """Recursively convert to JSON-safe types; drop image arrays."""
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _analysis_fields(res: Any) -> dict:
    """All SceneAnalysis fields except image arrays (getattr-tolerant)."""
    return {
        "scene": getattr(res, "scene", ""),
        "alert_level": getattr(res, "alert_level", "unknown"),
        "hazards": _jsonable(getattr(res, "hazards", []) or []),
        "actors": _jsonable(getattr(res, "actors", []) or []),
        "profile": getattr(res, "profile", ""),
        "latency_s": float(getattr(res, "latency_s", 0.0) or 0.0),
        "timings": _jsonable(getattr(res, "timings", {}) or {}),
        "cached": bool(getattr(res, "cached", False)),
    }


def _wait_bitchat_drain(client: Any, timeout_s: float = 30.0) -> bool:
    """Wait for the Bitchat client to finish sending. Returns True if drained."""
    for name in ("drain", "flush", "wait_idle", "wait_until_idle", "join"):
        fn = getattr(client, name, None)
        if callable(fn):
            try:
                r = fn(timeout=timeout_s)
                return True if r is None else bool(r)
            except TypeError:
                try:
                    r = fn(timeout_s)
                    return True if r is None else bool(r)
                except Exception:
                    pass
            except Exception as e:
                print(f"[bitchat] WARN: {name}() failed: {e}")
    # Generic fallback: poll the internal queue, then allow the in-flight item
    # (already popped by the worker) to finish.
    t_end = time.perf_counter() + timeout_s
    q = getattr(client, "_queue", None)
    lock = getattr(client, "_lock", None)
    while time.perf_counter() < t_end:
        if q is None:
            break
        if lock is not None:
            with lock:
                n = len(q)
        else:
            n = len(q)
        if n == 0:
            break
        time.sleep(0.2)
    else:
        return False
    # in-flight item: wait for a "sent" timestamp change or a short grace period
    last0 = getattr(client, "_last_sent_t", None)
    grace_end = min(t_end, time.perf_counter() + 12.0)
    while time.perf_counter() < grace_end:
        busy = getattr(client, "_busy", None) or getattr(client, "_sending", None)
        if busy is False:
            return True
        if last0 is not None and getattr(client, "_last_sent_t", last0) != last0 and busy is None:
            # something got sent; give one more beat in case it was the last item
            time.sleep(0.5)
            return True
        time.sleep(0.2)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline flood video understanding (VLM keyframes).")
    p.add_argument("video", help="path to a video file")
    p.add_argument("--profile", default="flood", help="VLM analysis profile (default: flood)")
    p.add_argument("--interval", type=float, default=2.0,
                   help="seconds of video time between periodic keyframes (default 2.0)")
    p.add_argument("--max-keyframes", type=int, default=8, help="hard cap on VLM calls (default 8)")
    p.add_argument("--bitchat", action="store_true",
                   help="send ONE summary message + highest-alert keyframe to Bitchat at the end")
    p.add_argument("--show", action="store_true", help="show marked keyframes as they are produced")
    p.add_argument("--out", default="data/flood_reports", help="report root dir (default data/flood_reports)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_wall0 = time.perf_counter()

    video_path = Path(args.video).expanduser().resolve()
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = APP_DIR / out_root
    # model weight paths in configs are relative to the app dir
    os.chdir(APP_DIR)

    cap = cv2.VideoCapture(str(video_path))
    if not video_path.is_file() or not cap.isOpened():
        print(f"[flood-video] ERROR: cannot open video: {video_path}", file=sys.stderr)
        cap.release()
        return 2
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
        print("[flood-video] WARN: video has no FPS metadata; assuming 30 fps for timestamps")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = n_frames / fps if n_frames > 0 else 0.0
    max_kf = max(1, int(args.max_keyframes))
    interval = float(args.interval)
    if duration > 0 and duration / max_kf > interval:
        interval = duration / max_kf
    print(f"[flood-video] {video_path.name}: {vw}x{vh} @ {fps:.2f} fps, "
          f"{n_frames} frames ({duration:.1f}s)  keyframe interval={interval:.2f}s  max={max_kf}")

    # --- VLM FIRST: its loader has the highest host-memory peak (Windows pagefile) ---
    from vlm.core import VLMConfig, VLMCore
    from vlm.analyzer import TrackMotionHistory, actors_from_tracks
    vlm_core = VLMCore(VLMConfig.from_yaml())
    if vw > 0 and vh > 0 and hasattr(vlm_core, "set_frame_size"):
        vlm_core.set_frame_size(vw, vh)
    if not vlm_core.initialize():
        print("[flood-video] ERROR: VLM failed to initialize", file=sys.stderr)
        cap.release()
        return 3
    analyzer = vlm_core.analyzer
    analyzer.set_profile(args.profile)
    if hasattr(analyzer, "warmup"):
        analyzer.warmup()
    print(f"[flood-video] VLM ready  profile={getattr(analyzer, 'profile_name', args.profile)}")

    # --- detector + tracker + flood heuristic only ---
    from core.config import load_models_config
    from core.detector import Detector
    from core.tracker import Tracker
    from core.events import FloodDetector
    detector = Detector()
    tracker = Tracker(detector)
    flood_det = FloodDetector((load_models_config() or {}).get("flood", {}) or {})
    history = TrackMotionHistory()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"{video_path.stem}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    keyframes: list[dict] = []          # report entries (no arrays)
    timeline: list[tuple[float, Any]] = []
    kf_images: list[np.ndarray] = []
    actor_tracks: dict[int, dict] = {}  # track_id -> {kind, observations}
    seen_ids_by_cls: dict[int, set] = {}
    last_flood: Any = None
    last_flood_t: float | None = None
    flood_events: list[dict] = []
    last_kf_t: float | None = None
    last_kf_thumb: np.ndarray | None = None
    vlm_latencies: list[float] = []
    win = "flood-video keyframes"

    def _spare_budget(t_now: float) -> int:
        """Keyframes left after reserving one per remaining periodic interval.

        Scene-change keyframes only spend spare budget, so a moving camera or
        churning water can't use up --max-keyframes in the first seconds.
        """
        reserved = int(max(0.0, duration - t_now) // interval) if duration > 0 else 0
        return max_kf - len(keyframes) - reserved

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = frame_idx / fps
            tracks = tracker.update(frame) or []
            history.update(tracks, t)
            for tr in tracks:
                if getattr(tr, "track_id", -1) >= 0:
                    seen_ids_by_cls.setdefault(int(tr.cls), set()).add(int(tr.track_id))

            flood_onset = False
            if frame_idx % _FLOOD_EVERY == 0:
                for ev in flood_det.detect(frame, frame_idx):
                    if ev.event_type == "FLOOD":
                        # FloodDetector re-fires every few seconds while water stays
                        # in view; only a first event (or one after a long quiet gap)
                        # is an onset worth an immediate keyframe.
                        if last_flood_t is None or t - last_flood_t > _FLOOD_ONSET_GAP_S:
                            flood_onset = True
                        last_flood_t = t
                        last_flood = ev
                        flood_events.append({"t": round(t, 2), "frame_idx": frame_idx,
                                             **_jsonable(ev.details)})
                        print(f"[flood-video] FLOOD sensor t={t:.1f}s {ev.details}")

            # --- cheap keyframe selection ---
            trigger = None
            if len(keyframes) < max_kf and t >= _FIRST_KEYFRAME_S:
                thumb = _thumb(frame)
                if last_kf_t is None:
                    trigger = "first"
                elif flood_onset:
                    trigger = "flood_onset"
                elif t - last_kf_t >= interval:
                    trigger = "interval"
                elif (last_kf_thumb is not None
                      and t - last_kf_t >= max(_SCENE_CHANGE_MIN_GAP_S, interval / 2.0)
                      and _spare_budget(t) > 0
                      and float(np.mean(np.abs(thumb - last_kf_thumb))) > _SCENE_CHANGE_DIFF):
                    trigger = "scene_change"

            if trigger is not None:
                actors = actors_from_tracks(tracks, history, frame.shape)
                extra = ""
                if last_flood is not None:
                    d = last_flood.details
                    extra = (f"Water sensor: moving water over "
                             f"{int(round(100 * d.get('water_ratio', 0)))}% of view, "
                             f"{int(round(100 * d.get('motion_ratio', 0)))}% of it moving")
                res = analyzer.analyze(frame, actors=actors, extra_context=extra)
                kidx = len(keyframes)
                fields = _analysis_fields(res)
                vlm_latencies.append(fields["latency_s"])
                id_map = {str(a.id): int(a.track_id) for a in actors}
                motion_by_id = {str(a.id): a.motion for a in actors}

                marked = getattr(res, "marked_frame", None)
                img = marked if marked is not None else frame
                img_name = f"keyframe_{kidx:02d}_t{t:.1f}.jpg"
                cv2.imwrite(str(out_dir / img_name), img)
                kf_images.append(img.copy())

                for a in fields["actors"]:
                    aid = str(a.get("id", ""))
                    tid = id_map.get(aid)
                    if tid is None:
                        continue
                    rec = actor_tracks.setdefault(tid, {"track_id": tid,
                                                        "kind": a.get("kind", ""),
                                                        "observations": []})
                    rec["observations"].append({
                        "t": round(t, 2), "actor_id": aid,
                        "state": a.get("state", ""), "action": a.get("action", ""),
                        "risk": a.get("risk", ""), "motion": motion_by_id.get(aid, ""),
                    })

                keyframes.append({
                    "index": kidx, "t": round(t, 2), "frame_idx": frame_idx,
                    "trigger": trigger, "image": img_name, "extra_context": extra,
                    "actor_map": id_map,
                    "input_actors": [{"id": str(a.id), "kind": a.kind, "track_id": int(a.track_id),
                                      "xyxy": [int(v) for v in a.xyxy], "motion": a.motion}
                                     for a in actors],
                    **fields,
                })
                timeline.append((t, res))
                last_kf_t = t
                last_kf_thumb = _thumb(frame)

                tm = fields["timings"]
                actor_txt = " | ".join(_actor_line(a) for a in fields["actors"]) or "no actors"
                print(f"t={t:5.1f}s [{str(fields['alert_level']).upper()}] {fields['scene']} | "
                      f"{actor_txt} | latency={fields['latency_s']:.2f}s "
                      f"tokens={tm.get('image_tokens', '?')} kv_reused={tm.get('kv_reused', '?')} "
                      f"cached={fields['cached']}  ({trigger})")

                if args.show:
                    cv2.imshow(win, img)
                    cv2.waitKey(1)
            frame_idx += 1
    finally:
        cap.release()

    if frame_idx == 0:
        print(f"[flood-video] ERROR: no frames could be read from {video_path}", file=sys.stderr)
        return 2

    # --- overall summary ---
    overall_text = ""
    if timeline:
        try:
            overall_text = analyzer.summarize_timeline(timeline)
        except Exception as e:
            print(f"[flood-video] WARN: summarize_timeline failed: {e}")
    overall_alert = "unknown"
    best_idx = None
    for i, kf in enumerate(keyframes):
        if best_idx is None or _alert_rank(kf["alert_level"]) > _alert_rank(keyframes[best_idx]["alert_level"]):
            best_idx = i
    if best_idx is not None:
        overall_alert = str(keyframes[best_idx]["alert_level"])

    actor_summary: list[dict] = []
    for tid in sorted(actor_tracks):
        rec = actor_tracks[tid]
        states = [f"t={o['t']:.1f}s {o['state'] or o['action'] or '?'}" for o in rec["observations"]]
        max_risk = max((o["risk"] for o in rec["observations"]), key=_alert_rank, default="")
        actor_summary.append({**rec, "timeline": states, "max_risk": max_risk})

    person_tracks = len(seen_ids_by_cls.get(0, set()))
    track_counts = {_COCO_SHORT.get(c, str(c)): len(ids) for c, ids in sorted(seen_ids_by_cls.items())}
    wall = time.perf_counter() - t_wall0
    mean_lat = float(np.mean(vlm_latencies)) if vlm_latencies else 0.0

    report = {
        "video": {"path": str(video_path), "fps": fps, "frames": n_frames, "frames_read": frame_idx,
                  "width": vw, "height": vh, "duration_s": round(duration, 2)},
        "profile": getattr(analyzer, "profile_name", args.profile),
        "settings": {"interval_s": round(interval, 3), "max_keyframes": max_kf,
                     "scene_change_diff": _SCENE_CHANGE_DIFF, "flood_every": _FLOOD_EVERY},
        "overall": {"alert_level": overall_alert, "summary": overall_text,
                    "highest_alert_keyframe": best_idx},
        "keyframes": keyframes,
        "actors": actor_summary,
        "track_counts": track_counts,
        "person_tracks": person_tracks,
        "flood_events": flood_events,
        "total_wall_s": round(wall, 2),
        "mean_vlm_latency_s": round(mean_lat, 3),
    }
    (out_dir / "report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")

    lines = [f"Flood video report: {video_path.name}",
             f"{vw}x{vh} @ {fps:.2f} fps, {duration:.1f}s, profile={report['profile']}",
             f"OVERALL ALERT: {overall_alert.upper()}",
             f"Summary: {overall_text or '(none)'}",
             f"Tracks: {track_counts}  (person tracks: {person_tracks})",
             f"Flood sensor events: {len(flood_events)}",
             "", "Timeline:"]
    for kf in keyframes:
        lines.append(f"  t={kf['t']:5.1f}s [{str(kf['alert_level']).upper()}] {kf['scene']}  "
                     f"({kf['trigger']}, {kf['latency_s']:.2f}s{', cached' if kf['cached'] else ''})")
        for a in kf["actors"]:
            lines.append(f"      {_actor_line(a)}")
    lines += ["", "Actors over time:"]
    for a in actor_summary:
        lines.append(f"  track {a['track_id']} ({a['kind']}, max risk {a['max_risk'] or '?'}): "
                     + "; ".join(a["timeline"]))
    lines += ["", f"Keyframes: {len(keyframes)}  mean VLM latency: {mean_lat:.2f}s  "
                  f"total wall time: {wall:.1f}s"]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[flood-video] report written to {out_dir}")

    # --- one Bitchat message at the end ---
    if args.bitchat:
        from core.bitchat import build_client_from_config
        from core.config import load_pipeline_config
        client = build_client_from_config(load_pipeline_config())
        if client is None:
            print("[flood-video] WARN: Bitchat disabled or not configured; nothing sent")
        elif best_idx is None:
            print("[flood-video] WARN: no keyframes analyzed; nothing sent to Bitchat")
        else:
            msg = f"[{overall_alert.upper()}] {video_path.name}: {overall_text}".strip()
            top = sorted(actor_summary, key=lambda a: _alert_rank(a["max_risk"]), reverse=True)[:4]
            for a in top:
                extra = f"\n#{a['track_id']} {a['kind']}: " + "; ".join(a["timeline"][-2:])
                if len(msg) + len(extra) > 700:
                    break
                msg += extra
            msg = msg[:700]
            client.send_scene(msg, kf_images[best_idx])
            drained = _wait_bitchat_drain(client, timeout_s=30.0)
            print(f"[flood-video] Bitchat message {'sent' if drained else 'still queued after 30s'}")

    if args.show:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
