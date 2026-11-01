"""Full frame-by-frame annotated breakdown of a video (side-by-side output).

Left panel  (every frame): YOLO+ByteTrack person/vehicle boxes with track ids,
            water regions from the HSV+motion water sensor, HUD (t, fps, VLM
            s/call), latest VLM scene + per-actor report, sensor verdict and a
            red "!!" line when the VLM and the sensor disagree.
Right panel (every frame): what the VLM was last asked to look at.
            Orange = image tiles that changed since the last full VLM call,
            blue = the crop actually sent. Header shows the call type:
              FULL CALL    whole frame (first call, big change, or every --full-every s)
              REGION CALL  only the changed crop, token budget scaled to its area
              SKIP         nothing changed -> previous answer reused, 0 tokens
Speed path used for every VLM call: vlm/analyzer.py (Set-of-Mark actor ids,
256-token budget, CUDA-graph decoder, KV cache reused across the two turns).

Outputs in data/annotated/<video>_<stamp>/:
  annotated.mp4    side-by-side video, same fps/length as the input
  frames.jsonl     one JSON line per frame (tracks, water boxes, sensor, VLM state)
  vlm_calls.json   every VLM call: type, region, tokens, latency, timings, report
  summary.txt      timeline + overall verdict

Usage:
  python annotate_video.py VIDEO [--profile flood] [--every 1.0] [--full-every 6]
                           [--show] [--out data/annotated]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

GREEN, BLUE, ORANGE = (0, 255, 0), (255, 160, 0), (0, 165, 255)
YELLOW, RED, WHITE, BLACK = (0, 255, 255), (60, 60, 255), (255, 255, 255), (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
CAP_H = 150                  # caption bar under the two panels
REGION_MAX_FRAC = 0.55       # region crop above this share of the frame -> full call
_COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
         8: "boat", 16: "dog", 17: "horse", 19: "cow"}


def label(img, text, org, color, scale=0.45, thick=1, bg=BLACK):
    (tw, th), b = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    y = max(th + 2, min(img.shape[0] - b - 1, y))
    x = max(0, min(img.shape[1] - tw - 2, x))
    if bg is not None:
        cv2.rectangle(img, (x, y - th - 3), (x + tw + 3, y + b), bg, -1)
    cv2.putText(img, text, (x + 1, y - 1), FONT, scale, color, thick, cv2.LINE_AA)


class WaterRegions:
    """Water boxes from the same HSV ranges as core.events.FloodDetector, with stable ids."""

    def __init__(self, flood_det, min_area_frac=0.03):
        self.fd = flood_det
        self.min_area = min_area_frac
        self.prev_gray = None
        self.tracks: dict[int, tuple] = {}
        self.next_id = 1

    def update(self, frame):
        h, w = frame.shape[:2]
        sw = 160
        sh = max(1, int(h * sw / w))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        water = cv2.inRange(hsv, self.fd.mud_low, self.fd.mud_high) | \
            cv2.inRange(hsv, self.fd.blue_low, self.fd.blue_high)
        moving = np.zeros_like(water)
        if self.prev_gray is not None:
            moving = ((cv2.absdiff(gray, self.prev_gray) > self.fd.motion_thresh) * 255).astype(np.uint8)
        self.prev_gray = gray
        # flowing water = water-coloured AND textured motion, dilated into regions
        flow = cv2.dilate(water & moving, np.ones((5, 5), np.uint8), iterations=2) & water
        flow = cv2.morphologyEx(flow, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(flow, 8)
        sx, sy = w / sw, h / sh
        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < self.min_area * sw * sh:
                continue
            fill = float(np.count_nonzero(moving[lab == i])) / max(1, area)
            boxes.append(((int(x * sx), int(y * sy), int((x + bw) * sx), int((y + bh) * sy)), fill))
        ratio = float(np.count_nonzero(water)) / water.size
        mratio = float(np.count_nonzero(water & moving)) / max(1, np.count_nonzero(water))
        return self._assign(boxes), ratio, mratio

    def _assign(self, boxes):
        def iou(a, b):
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / ua if ua else 0.0
        out, used = [], set()
        for box, conf in sorted(boxes, key=lambda b: -(b[0][2] - b[0][0]) * (b[0][3] - b[0][1])):
            best, best_iou = None, 0.3
            for tid, tb in self.tracks.items():
                if tid not in used and iou(box, tb) > best_iou:
                    best, best_iou = tid, iou(box, tb)
            if best is None:
                best = self.next_id
                self.next_id += 1
            used.add(best)
            self.tracks[best] = box
            out.append((best, box, conf))
        self.tracks = {k: v for k, v in self.tracks.items() if k in used}
        return out


def actor_changes(tracks, ref_actors, w, h, move_frac=0.6, pad=0.6):
    """Regions whose CONTENT changed since the reference call.

    Pixel diffs are useless on flood footage (moving camera + churning water
    change every tile every second), so change is semantic: an actor that is
    new since the reference call, or that moved more than `move_frac` of its
    own box height. Returns padded boxes around those actors.
    """
    boxes = []
    for tr in tracks:
        if tr.track_id < 0:
            continue
        x1, y1, x2, y2 = tr.xyxy
        cx, cy, bh = (x1 + x2) / 2, (y1 + y2) / 2, max(1, y2 - y1)
        ref = ref_actors.get(tr.track_id)
        if ref is not None and np.hypot(cx - ref[0], cy - ref[1]) < move_frac * bh:
            continue
        bw = x2 - x1
        px, py = int(bw * pad) + 12, int(bh * pad) + 12
        boxes.append((max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)))
    return boxes


def camera_shift(ref_gray, gray):
    (dx, dy), _ = cv2.phaseCorrelate(ref_gray.astype(np.float32), gray.astype(np.float32))
    return float(np.hypot(dx, dy)) / gray.shape[0]


def union_box(boxes, w, h, pad=0.04):
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    px, py = int(w * pad), int(h * pad)
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


def wrap_lines(text, width_px, scale):
    cw = max(1, int(cv2.getTextSize("M", FONT, scale, 1)[0][0] * 0.82))
    return textwrap.wrap(text, max(10, width_px // cw)) or [""]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Frame-by-frame annotated video breakdown")
    ap.add_argument("video")
    ap.add_argument("--profile", default="flood")
    ap.add_argument("--every", type=float, default=1.0, help="seconds of video between VLM decisions")
    ap.add_argument("--full-every", type=float, default=6.0, help="force a full-frame call this often (s)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--out", default="data/annotated")
    args = ap.parse_args(argv)

    video = Path(args.video).expanduser().resolve()
    os.chdir(APP_DIR)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[annotate] ERROR: cannot open {video}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[annotate] {video.name}: {W}x{H} @ {fps:.1f} fps, {n_frames} frames")

    # VLM first (largest host-memory peak), then the light models
    from vlm.core import VLMConfig, VLMCore
    from vlm.analyzer import TrackMotionHistory, actors_from_tracks, Actor
    core = VLMCore(VLMConfig.from_yaml())
    if not core.initialize():
        return 3
    an = core.analyzer
    an.set_profile(args.profile)
    an.cfg["skip_if_unchanged"] = False          # skipping is decided here, per region
    base_budget = int(an.cfg.get("image_max_pixels", 200704))
    from core.config import load_models_config
    from core.detector import Detector
    from core.tracker import Tracker
    from core.events import FloodDetector
    det = Detector()
    trk = Tracker(det)
    flood_det = FloodDetector((load_models_config() or {}).get("flood", {}) or {})
    water = WaterRegions(flood_det)
    hist = TrackMotionHistory()

    out_dir = APP_DIR / args.out / f"{video.stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    for cc in ("mp4v",):
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*cc), fps, (2 * W, H + CAP_H))
        if writer.isOpened():
            break
    jf = open(out_dir / "frames.jsonl", "w", encoding="utf-8")

    step = max(1, int(round(args.every * fps)))
    ref_gray = None              # frame of the last FULL call
    ref_wratio = 0.0
    ref_actors: dict[int, tuple] = {}
    last_full = None
    last_full_t = -1e9
    last = None                  # last SceneAnalysis shown
    panel = {"kind": "WAITING", "frac": 0.0, "tokens": 0, "tiles": [], "crop": None, "t": 0.0}
    calls, vlm_lat = [], []
    flood_on_t = -1e9
    from collections import deque
    perc_hist = deque(maxlen=30)
    proc_fps = 0.0
    fi = 0
    t_wall = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        t0 = time.perf_counter()
        tracks = trk.update(frame) or []
        hist.update(tracks, t)
        wboxes, wratio, mratio = water.update(frame)
        if fi % 5 == 0 and flood_det.detect(frame, fi):
            flood_on_t = t
        perc_s = time.perf_counter() - t0
        sensor_flood = (t - flood_on_t) <= 5.0 or (wratio >= flood_det.min_ratio and mratio >= flood_det.min_motion_ratio)
        sensor_txt = (f"sensor: {'FLOOD' if sensor_flood else 'no flood'}  water {wratio * 100:.0f}% "
                      f"of view, {mratio * 100:.0f}% moving")

        # ---- VLM decision: full / region / skip --------------------------------
        if fi % step == 0 and t >= 0.3:
            small = cv2.cvtColor(cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            extra = f"Water sensor: {wratio * 100:.0f}% of view is water, {mratio * 100:.0f}% of it moving."
            tiles = []
            reason = ""
            if ref_gray is None:
                kind, reason = "FULL CALL", "first"
            elif t - last_full_t >= args.full_every:
                kind, reason = "FULL CALL", f"refresh every {args.full_every:.0f}s"
            elif camera_shift(ref_gray, small) > 0.08:
                kind, reason = "FULL CALL", "camera moved"
            elif abs(wratio - ref_wratio) > 0.10:
                kind, reason = "FULL CALL", f"water extent {ref_wratio * 100:.0f}->{wratio * 100:.0f}%"
            else:
                tiles = actor_changes(tracks, ref_actors, W, H)
                if not tiles:
                    kind, reason = "SKIP", "no new/moved actors"
                else:
                    crop = union_box(tiles, W, H, pad=0.0)
                    # a crop the VLM can actually read: at least ~1/3 of each side
                    mw, mh = max(W // 3, 140), max(H // 3, 140)
                    cx, cy = (crop[0] + crop[2]) // 2, (crop[1] + crop[3]) // 2
                    if crop[2] - crop[0] < mw:
                        x1c = min(max(0, cx - mw // 2), W - mw); crop = (x1c, crop[1], x1c + mw, crop[3])
                    if crop[3] - crop[1] < mh:
                        y1c = min(max(0, cy - mh // 2), H - mh); crop = (crop[0], y1c, crop[2], y1c + mh)
                    frac = (crop[2] - crop[0]) * (crop[3] - crop[1]) / float(W * H)
                    if frac > REGION_MAX_FRAC:
                        kind, reason = "FULL CALL", "changes cover most of frame"
                    else:
                        kind, reason = "REGION CALL", f"{len(tiles)} new/moved actor(s)"
            if kind == "FULL CALL":
                crop = (0, 0, W, H)
            if kind == "SKIP":
                panel = {"kind": "SKIP (0 tok, answer kept)", "frac": 0.0, "tokens": 0,
                         "tiles": [], "crop": None, "t": t, "reason": reason}
                calls.append({"t": round(t, 2), "frame": fi, "type": "skip", "tokens": 0,
                              "latency_s": 0.0, "reason": reason})
            else:
                x1, y1, x2, y2 = crop
                sub = frame[y1:y2, x1:x2]
                frac_px = (x2 - x1) * (y2 - y1) / float(W * H)
                budget = base_budget if kind == "FULL CALL" else max(50176, int(base_budget * frac_px))
                inside = []
                for tr in tracks:
                    bx1, by1, bx2, by2 = tr.xyxy
                    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    if x1 <= cx < x2 and y1 <= cy < y2:
                        inside.append(tr)
                actors = actors_from_tracks(inside, hist, frame.shape)
                actors = [Actor(a.id, a.kind, a.track_id,
                                (max(0, a.xyxy[0] - x1), max(0, a.xyxy[1] - y1),
                                 min(x2 - x1 - 1, a.xyxy[2] - x1), min(y2 - y1 - 1, a.xyxy[3] - y1)),
                                a.motion) for a in actors]
                res = an.analyze(sub, actors=actors, extra_context=extra, force=True, max_pixels=budget)
                tm = res.timings
                vlm_lat.append(res.latency_s)
                if kind == "FULL CALL":
                    ref_gray, last_full_t, ref_wratio = small, t, wratio
                # actors analysed now become the reference for "moved/new"
                for tr in (tracks if kind == "FULL CALL" else inside):
                    if tr.track_id >= 0:
                        bx1, by1, bx2, by2 = tr.xyxy
                        ref_actors[tr.track_id] = ((bx1 + bx2) / 2, (by1 + by2) / 2)
                if kind == "REGION CALL" and last is not None and last_full is not None:
                    # region answer updates the actors it covered; keep the full-frame scene
                    res.scene = "[region update] " + res.scene + "  (full view: " + last_full.scene[:120] + ")"
                if kind == "FULL CALL":
                    last_full = res
                last = res
                panel = {"kind": kind, "frac": frac_px, "tokens": tm["image_tokens"], "tiles": tiles,
                         "reason": reason,
                         "crop": crop, "t": t, "lat": res.latency_s, "kv": tm.get("kv_reused"),
                         "dec": tm.get("decoder"), "tps": tm.get("decode_tok_per_s")}
                calls.append({"t": round(t, 2), "frame": fi, "type": kind.split()[0].lower(), "reason": reason,
                              "crop": list(crop), "area_pct": round(100 * frac_px, 1),
                              "tokens": tm["image_tokens"], "latency_s": round(res.latency_s, 2),
                              "kv_reused": tm.get("kv_reused"), "decoder": tm.get("decoder"),
                              "decode_tok_per_s": tm.get("decode_tok_per_s"),
                              "scene": res.scene, "alert_level": res.alert_level,
                              "hazards": res.hazards, "actors": res.actors,
                              "actor_track_ids": {a.id: a.track_id for a in actors}})
                print(f"t={t:5.1f}s {kind:<11} {100 * frac_px:5.1f}% px {tm['image_tokens']:>3} tok "
                      f"{res.latency_s:.2f}s kv_reused={tm.get('kv_reused')} {tm.get('decoder')} "
                      f"[{res.alert_level}] {res.scene[:90]}")

        # ---- left panel -------------------------------------------------------
        left = frame.copy()
        for wid, (bx1, by1, bx2, by2), conf in wboxes:
            cv2.rectangle(left, (bx1, by1), (bx2, by2), BLUE, 2)
            label(left, f"#W{wid} {'flood water' if sensor_flood else 'water'} {conf:.2f}",
                  (bx1 + 4, max(by1, 22) + 16), BLUE)
        for tr in tracks:
            if tr.track_id < 0:
                continue
            bx1, by1, bx2, by2 = tr.xyxy
            cv2.rectangle(left, (bx1, by1), (bx2, by2), GREEN, 2)
            label(left, f"#{tr.track_id} {_COCO.get(tr.cls, tr.cls)} {tr.conf:.2f}", (bx1, by2 + 14), GREEN)
        perc_hist.append(perc_s)
        proc_fps = len(perc_hist) / max(1e-6, sum(perc_hist))
        vsf = (sum(vlm_lat) / len(vlm_lat)) if vlm_lat else 0.0
        n_real = sum(1 for c in calls if c["type"] != "skip")
        label(left, f"t={t:4.1f}s det fps={proc_fps:4.1f} VLM s/call={vsf:.1f} calls={n_real}",
              (2, 14), YELLOW, 0.45, 1)
        bottom = []
        if last is not None:
            bottom += [("SCENE: " + last.scene, WHITE)]
            bottom += [(f"[{last.alert_level.upper()}] " + "; ".join(
                f"{a['id']} {a.get('state', '')} {a.get('action', '')} risk:{a.get('risk', '')}"
                for a in last.actors[:4]), WHITE)]
        bottom.append((sensor_txt, YELLOW))
        if last is not None:
            vlm_flood = any(h.get("type") in ("flood", "fast_current", "rising_water", "road_flooded",
                                              "bridge_submerged") for h in last.hazards) or \
                last.alert_level in ("high", "critical")
            if vlm_flood and not sensor_flood:
                bottom.append(("!! VLM says 'flood' but sensor sees no moving water", RED))
            elif sensor_flood and not vlm_flood:
                bottom.append((f"!! sensor says FLOOD but VLM says '{last.alert_level}'", RED))
        # ---- right panel ------------------------------------------------------
        right = frame.copy()
        for tb in panel["tiles"]:
            cv2.rectangle(right, tb[:2], tb[2:], ORANGE, 2)
        if panel["crop"] is not None:
            cx1, cy1, cx2, cy2 = panel["crop"]
            cv2.rectangle(right, (cx1 + 1, cy1 + 1), (cx2 - 2, cy2 - 2), BLUE, 3)
        hdr = panel["kind"]
        if panel["kind"] in ("FULL CALL", "REGION CALL"):
            hdr += f": {panel['frac'] * 100:.1f}% px  {panel['tokens']} tok"
        label(right, hdr, (2, 14), YELLOW, 0.48, 1)
        if "lat" in panel:
            label(right, f"@t={panel['t']:.1f}s {panel['lat']:.1f}s  kv_reuse={panel['kv']}  "
                         f"{panel['dec']} {panel['tps'] or '-'} tok/s", (2, 32), YELLOW, 0.4, 1)
        if panel.get("reason"):
            label(right, "why: " + panel["reason"], (2, 50), YELLOW, 0.4, 1)
        label(right, "orange=new/moved actors  blue=crop sent to VLM", (2, H - 6), WHITE, 0.38)

        cap_bar = np.zeros((CAP_H, 2 * W, 3), np.uint8)
        lines = []
        for k, (text, col) in enumerate(bottom):
            for ln in wrap_lines(text, 2 * W - 10, 0.45)[:(3 if k == 0 else 2)]:
                lines.append((ln, col))
        for k, (ln, col) in enumerate(lines[:CAP_H // 17]):
            cv2.putText(cap_bar, ln, (5, 16 + 17 * k), FONT, 0.45, col, 1, cv2.LINE_AA)
        out = np.vstack([np.hstack([left, right]), cap_bar])
        writer.write(out)
        jf.write(json.dumps({
            "frame": fi, "t": round(t, 3),
            "tracks": [{"id": tr.track_id, "cls": _COCO.get(tr.cls, tr.cls), "conf": round(tr.conf, 3),
                        "xyxy": list(map(int, tr.xyxy)), "motion": hist.describe(tr.track_id)}
                       for tr in tracks if tr.track_id >= 0],
            "water": [{"id": wid, "xyxy": list(b), "moving": round(c, 3)} for wid, b, c in wboxes],
            "sensor": {"flood": bool(sensor_flood), "water_ratio": round(wratio, 3), "moving_ratio": round(mratio, 3)},
            "vlm_panel": panel["kind"],
            "vlm_alert": last.alert_level if last is not None else None,
        }) + "\n")
        if args.show:
            cv2.imshow("annotate", out)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        fi += 1
    cap.release()
    writer.release()
    jf.close()
    if args.show:
        cv2.destroyAllWindows()

    real = [c for c in calls if c["type"] != "skip"]
    from vlm.analyzer import SceneAnalysis
    timeline = [(c["t"], SceneAnalysis(c["scene"], c["alert_level"], c["hazards"], c["actors"],
                                       args.profile, c["latency_s"], {})) for c in real]
    overall = an.summarize_timeline(timeline) if timeline else ""
    (out_dir / "vlm_calls.json").write_text(json.dumps(calls, indent=2), encoding="utf-8")
    full_tok = real and max(c["tokens"] for c in real if c["type"] == "full")
    spent = sum(c["tokens"] for c in calls)
    naive = len(calls) * (full_tok or 0)
    wall = time.perf_counter() - t_wall
    lines = [f"Annotated breakdown: {video.name} ({W}x{H}, {fps:.0f} fps, {fi} frames, {fi / fps:.1f}s)",
             f"VLM decisions: {len(calls)}  full={sum(c['type'] == 'full' for c in calls)}  "
             f"region={sum(c['type'] == 'region' for c in calls)}  skip={sum(c['type'] == 'skip' for c in calls)}",
             f"image tokens spent: {spent}  vs {naive} if every decision were a full call "
             f"({100 * (1 - spent / naive):.0f}% saved)" if naive else "",
             f"mean VLM latency: {np.mean([c['latency_s'] for c in real]):.2f}s  "
             f"kv_reused on all calls: {all(c['kv_reused'] for c in real)}" if real else "",
             f"total wall time: {wall:.1f}s", "", "OVERALL: " + overall, "", "Timeline:"]
    for c in calls:
        if c["type"] == "skip":
            lines.append(f"  t={c['t']:5.1f}s SKIP   0 tok ({c.get('reason', '')}; previous answer kept)")
            continue
        lines.append(f"  t={c['t']:5.1f}s {c['type'].upper():<6} {c['area_pct']:5.1f}% px {c['tokens']:>3} tok "
                     f"({c.get('reason', '')}) "
                     f"{c['latency_s']:.2f}s [{c['alert_level']}] {c['scene']}")
        for a in c["actors"]:
            lines.append(f"      {a['id']} (track {c['actor_track_ids'].get(a['id'], '?')}): "
                         f"{a.get('state')} / {a.get('action')} / risk {a.get('risk')}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[annotate] video: {out_dir / 'annotated.mp4'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
