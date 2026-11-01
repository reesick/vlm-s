"""VLM speed benchmark on a real video.

Usage (from the app dir):
    .venv\\Scripts\\python.exe benchmarks\\vlm_speed_bench.py --frames 3 --profile flood [--legacy]

Modes, all on the same N evenly spaced frames:
  legacy        old VLMCore path: two independent HF generate() calls, each
                re-encoding the image at model.max_pixels (602112, ~768 tokens).
  hf_reuse      analyzer on HF generate(): 256-token image budget, Set-of-Mark
                actors, turn 2 continues turn 1's DynamicCache, prompt lookup.
  graph_noreuse analyzer on the CUDA-graph decoder, turn 2 re-prefills the
                whole conversation (image re-encoded).
  graph_reuse   analyzer on the CUDA-graph decoder, turn 2 appended to the same
                static KV cache (default production path).
  graph_blur    graph_reuse + focus blur (background outside actor boxes blurred).

graph_noreuse and graph_reuse feed the model the same tokens, so their greedy
outputs should be identical; that is the KV-reuse correctness check.
Actors come from YOLOv8n + ByteTrack (CPU) over the second before each frame.
Marked frames are written to benchmarks/out/.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import cv2  # noqa: E402

DEFAULT_VIDEO = "WhatsApp Video 2026-10-10 at 11.34.09 AM.mp4"


def resolve_video(arg: str | None) -> Path:
    name = arg or DEFAULT_VIDEO
    p = Path(name)
    cands = [p] if p.is_absolute() else [Path.cwd() / p, APP_DIR / p, APP_DIR.parent / p,
                                         APP_DIR.parents[1] / p, APP_DIR.parents[1] / p.name]
    for c in cands:
        if c.is_file():
            return c.resolve()
    raise SystemExit(f"video not found: {name!r}")


class _T:
    def __init__(self, tid, cls, conf, xyxy):
        self.track_id, self.cls, self.conf, self.xyxy = tid, cls, conf, xyxy


def sample_frames(video: Path, n: int, use_actors: bool, max_actors: int):
    from vlm.analyzer import TrackMotionHistory, actors_from_tracks
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idxs = [int(total * (i + 1) / (n + 1)) for i in range(n)]
    yolo = None
    if use_actors:
        from ultralytics import YOLO
        yolo = YOLO(str(APP_DIR / "models" / "yolov8n.pt"))
    out = []
    for idx in idxs:
        actors = None
        if yolo is not None:
            hist = TrackMotionHistory()
            step = max(1, int(round(fps * 0.2)))
            tracks, fr = [], None
            for k, fi in enumerate(range(max(0, idx - 5 * step), idx + 1, step)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, fr = cap.read()
                if not ok:
                    continue
                r = yolo.track(fr, persist=(k > 0), device="cpu", verbose=False,
                               classes=[0, 1, 2, 3, 5, 7, 8], conf=0.3, tracker="bytetrack.yaml")[0]
                tracks = []
                if r.boxes is not None and r.boxes.id is not None:
                    for b, tid, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                             r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                        tracks.append(_T(tid, c, cf, tuple(int(v) for v in b)))
                hist.update(tracks, t=fi / fps)
            actors = actors_from_tracks(tracks, hist, fr.shape, max_actors)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        out.append((idx, idx / fps, frame, actors))
    cap.release()
    return out


def jpeg_kb(img) -> float:
    h, w = img.shape[:2]
    s = min(1.0, 640 / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return len(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]) / 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--profile", default="flood")
    ap.add_argument("--no-actors", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="also run the old two-pass path (slow)")
    ap.add_argument("--skip-hf", action="store_true", help="skip the hf_reuse mode")
    args = ap.parse_args()

    import torch
    from vlm.core import VLMCore, VLMConfig
    from vlm.analyzer import VLMAnalyzer, parse_json_object

    video = resolve_video(args.video)
    cfg = VLMConfig.from_yaml()
    frames = sample_frames(video, args.frames, not args.no_actors,
                           int(cfg.analyzer_cfg.get("max_actors", 8)))
    print(f"[bench] video={video}")
    for idx, t, _, actors in frames:
        print(f"[bench] frame {idx} t={t:.1f}s actors="
              + (", ".join(f"{a.id}:{a.motion}" for a in actors) if actors else "none"))

    cfg.analyzer_cfg = {**cfg.analyzer_cfg, "warmup": False}
    core = VLMCore(cfg)
    if not core.initialize():
        raise SystemExit("[bench] VLM failed to initialize")
    model, proc = core._model, core._processor
    base = {**cfg.analyzer_cfg, "skip_if_unchanged": False, "warmup": True}
    modes = {}
    if not args.skip_hf:
        modes["hf_reuse"] = {**base, "fast_decode": False, "kv_reuse": True}
    modes["graph_noreuse"] = {**base, "fast_decode": True, "kv_reuse": False}
    modes["graph_reuse"] = {**base, "fast_decode": True, "kv_reuse": True}
    modes["graph_blur"] = {**base, "fast_decode": True, "kv_reuse": True, "focus_blur": True}
    analyzers = {m: VLMAnalyzer(model, proc, c, cfg.profiles, args.profile) for m, c in modes.items()}

    out_dir = APP_DIR / "benchmarks" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {m: {"lat": [], "t1": [], "t2": [], "tps": [], "vram": [], "json_ok": [], "kb": []}
             for m in ["legacy", *modes]}
    results = {m: [] for m in modes}

    def timed(fn):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter(); r = fn(); torch.cuda.synchronize()
        return r, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**20

    for idx, t, frame, actors in frames:
        print(f"\n===== frame {idx} (t={t:.1f}s) =====")
        if args.legacy:
            ents, dt, vr = timed(lambda: core._run_model_inference_legacy(frame))
            stats["legacy"]["lat"].append(dt); stats["legacy"]["vram"].append(vr)
            print(f"[legacy] {dt:.2f}s scene={core._last_scene_description!r}")
        for m, an in analyzers.items():
            a, dt, vr = timed(lambda: an.analyze(frame, actors))
            tm = a.timings
            s = stats[m]
            s["lat"].append(dt); s["vram"].append(vr); s["t1"].append(tm["turn1_s"]); s["t2"].append(tm["turn2_s"])
            s["json_ok"].append(parse_json_object(tm["raw_report"]) is not None)
            if tm.get("decode_tok_per_s"):
                s["tps"].append(tm["decode_tok_per_s"])
            img = a.marked_frame if a.marked_frame is not None else frame
            s["kb"].append(jpeg_kb(img))
            results[m].append((t, a))
            cv2.imwrite(str(out_dir / f"frame_{idx:06d}_{m}.jpg"), img)
            print(f"[{m}] {dt:.2f}s decoder={tm.get('decoder')} turn1={tm['turn1_s']}s "
                  f"turn2={tm['turn2_s']}s tok/s={tm.get('decode_tok_per_s', '-')} "
                  f"img_tok={tm['image_tokens']} kv_reused={tm['kv_reused']} "
                  f"new_tok={tm.get('turn1_new_tokens')}+{tm.get('turn2_new_tokens')}")
            print(f"    scene: {a.scene}")
            print(f"    json : {tm['raw_report']}")

    print("\n================ RESULTS ================")
    print(f"{'mode':<15}{'mean s':>8}{'turn1':>8}{'turn2':>8}{'tok/s':>8}{'VRAM MiB':>10}{'JSON ok':>9}{'jpeg KB':>9}")
    for m, s in stats.items():
        if not s["lat"]:
            continue
        f = lambda v: f"{statistics.mean(v):.2f}" if v else "-"  # noqa: E731
        tps = f"{statistics.mean(s['tps']):.1f}" if s["tps"] else "-"
        jok = f"{sum(s['json_ok'])}/{len(s['json_ok'])}" if s["json_ok"] else "-"
        print(f"{m:<15}{f(s['lat']):>8}{f(s['t1']):>8}{f(s['t2']):>8}{tps:>8}"
              f"{max(s['vram']):>10.0f}{jok:>9}{f(s['kb']):>9}")
    ref = "legacy" if stats["legacy"]["lat"] else ("hf_reuse" if "hf_reuse" in stats and stats["hf_reuse"]["lat"] else None)
    if ref:
        r = statistics.mean(stats[ref]["lat"])
        print("speedup vs " + ref + ": " + ", ".join(
            f"{m} x{r / statistics.mean(stats[m]['lat']):.2f}" for m in modes if m != ref))

    def same(m1, m2):
        return all(a.scene == b.scene and a.timings["raw_report"] == b.timings["raw_report"]
                   for (_, a), (_, b) in zip(results[m1], results[m2]))
    print(f"KV reuse correctness (graph_reuse == graph_noreuse outputs): {same('graph_reuse', 'graph_noreuse')}")
    if "hf_reuse" in results:
        print(f"graph decoder == HF generate outputs: {same('graph_reuse', 'hf_reuse')}")

    an = analyzers["graph_reuse"]
    t0 = time.perf_counter()
    verdict = an.summarize_timeline(results["graph_reuse"])
    print(f"\nOverall verdict (text-only, {time.perf_counter() - t0:.2f}s): {verdict}")
    print("\n--- to_message() of last frame ---\n" + results["graph_reuse"][-1][1].to_message())
    print(f"\nmarked frames saved to {out_dir}")


if __name__ == "__main__":
    main()
