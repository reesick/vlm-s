"""Quantization accuracy + efficiency benchmark for Qwen2.5-VL-3B on flood footage.

Same model, different weight formats. Accuracy is measured two ways:
  1. Fidelity to full precision: an fp16 reference (partly CPU-offloaded, slow,
     run once and cached) produces the scene text for each frame. Each variant
     is teacher-forced on that text; we report top-1 token agreement and the
     NLL of the reference tokens (lower = closer to fp16).
  2. Task accuracy against hand labels of the clip (labels.json): alert level
     high/critical, flood + fast current reported, people counted.
Efficiency: load VRAM, peak VRAM, end-to-end analyze() latency, decode tok/s.

Each variant runs in its own process (clean VRAM):
  python benchmarks/quant_eval.py --variant fp16_ref     # once, ~10-20 min
  python benchmarks/quant_eval.py --variant nf4
  python benchmarks/quant_eval.py --report
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

OUT = APP_DIR / "benchmarks" / "out" / "quant"
VIDEO = APP_DIR.parents[1] / "WhatsApp Video 2026-10-10 at 11.34.09 AM.mp4"
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
FRAME_IDX = [20, 60, 100, 140, 180, 220, 260, 300]
HELDOUT_FROM = 180            # tools/qlora_train.py trains on frames 0-169 only
NEG_EVAL = ["data/keyframes/FALL_2026-10-30T12-32-45_f2082_id208.jpg"]   # unseen, no water

VARIANTS = {
    # name: (description, loader kwargs builder)
    "fp16_ref": "fp16 weights, GPU+CPU offload (reference, not deployable on 6 GB)",
    "nf4": "current: NF4 4-bit all linears (vision + LLM), fp16 compute",
    "nf4_dq": "NF4 + double quantization (QLoRA default), fp16 compute",
    "nf4_vision16": "NF4 LLM, vision encoder kept fp16",
    "int8": "bitsandbytes LLM.int8 8-bit",
    "int8_vision16": "LLM.int8 LLM, vision encoder kept fp16",
}


def load_variant(name: str, adapter: str | None = None):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    kw = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True,
              device_map="auto", offload_folder="D:/qwen_vlm_offload")
    if name == "fp16_ref":
        kw["max_memory"] = {0: "4GiB", "cpu": "14GiB"}
    else:
        kw["max_memory"] = {0: "5GiB", "cpu": "5GiB"}
        skip = ["lm_head"] + (["visual"] if name.endswith("vision16") else [])
        if name.startswith("nf4"):
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_use_double_quant=(name == "nf4_dq"),
                                   llm_int8_skip_modules=skip)
        elif name.startswith("int8"):
            q = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
        else:
            raise SystemExit(f"unknown variant {name}")
        kw["quantization_config"] = q
    torch.cuda.reset_peak_memory_stats()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, **kw).eval()
    if adapter:
        from peft import PeftModel
        # merged into the 4-bit weights, exactly as vlm/core.py deploys it
        model = PeftModel.from_pretrained(model, adapter)
        model = (model.eval() if os.environ.get("LORA_NOMERGE") else model.merge_and_unload().eval())
    proc = AutoProcessor.from_pretrained(MODEL)
    load_gb = torch.cuda.memory_allocated() / 2**30
    dm = getattr(model, "hf_device_map", {}) or {}
    offloaded = sorted({str(v) for v in dm.values()} - {"0", "cuda:0"})
    return model, proc, load_gb, offloaded


def eval_frames():
    """Fixed frames + YOLO/ByteTrack actors from the second before each frame."""
    from vlm.analyzer import TrackMotionHistory, actors_from_tracks
    from ultralytics import YOLO
    yolo = YOLO(str(APP_DIR / "models" / "yolov8n.pt"))
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []

    class _T:
        def __init__(s, tid, cls, conf, xyxy):
            s.track_id, s.cls, s.conf, s.xyxy = tid, cls, conf, xyxy
    for idx in FRAME_IDX:
        hist, tracks, fr = TrackMotionHistory(), [], None
        step = 6
        for k, fi in enumerate(range(max(0, idx - 5 * step), idx + 1, step)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            r = yolo.track(fr, persist=(k > 0), device="cpu", verbose=False,
                           classes=[0, 1, 2, 3, 5, 7, 8], conf=0.3, tracker="bytetrack.yaml")[0]
            tracks = []
            if r.boxes is not None and r.boxes.id is not None:
                for b, tid, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                         r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                    tracks.append(_T(tid, c, cf, tuple(int(v) for v in b)))
            hist.update(tracks, t=fi / fps)
        out.append((idx, idx / fps, fr, actors_from_tracks(tracks, hist, fr.shape)))
    cap.release()
    return out


def neg_frames():
    out = []
    for pth in NEG_EVAL:
        img = cv2.imread(str(APP_DIR / pth))
        if img is not None:
            out.append((pth, img))
    return out


def scene_inputs(an, frame, actors):
    from vlm.analyzer import budget_size, draw_actor_marks, resize_for_budget
    budget = int(an.cfg.get("image_max_pixels", 200704))
    h, w = frame.shape[:2]
    nw, _ = budget_size(w, h, budget)
    marked = bool(actors)
    src = draw_actor_marks(frame, actors, nw / float(w)) if marked else frame
    rgb = cv2.cvtColor(resize_for_budget(src, budget), cv2.COLOR_BGR2RGB)
    return an._encode(an._scene_prompt(actors, marked), rgb)


def teacher_forced(an, model, frame, actors, ref_text):
    """Top-1 agreement and mean NLL of the fp16 reference scene text under this model."""
    import torch
    inputs = scene_inputs(an, frame, actors)
    tgt = an.tokenizer(ref_text + "<|im_end|>", add_special_tokens=False,
                       return_tensors="pt").input_ids.to(inputs["input_ids"].device)
    full = torch.cat([inputs["input_ids"], tgt], dim=1)
    L = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        model.model.rope_deltas = None
        logits = model(**an._fresh_inputs(inputs, full)).logits[0, L - 1:-1].float()
    t = tgt[0]
    agree = float((logits.argmax(-1) == t).float().mean())
    nll = float(torch.nn.functional.cross_entropy(logits, t))
    return agree, nll


def run_variant(name: str, adapter: str | None, tag: str) -> None:
    import torch
    from vlm.analyzer import VLMAnalyzer
    from vlm.core import VLMConfig
    OUT.mkdir(parents=True, exist_ok=True)
    frames = eval_frames()
    cfg = VLMConfig.from_yaml()
    t0 = time.perf_counter()
    model, proc, load_gb, offloaded = load_variant(name, adapter)
    load_s = time.perf_counter() - t0
    fast = name != "fp16_ref"
    an = VLMAnalyzer(model, proc, {**cfg.analyzer_cfg, "skip_if_unchanged": False,
                                   "fast_decode": fast, "warmup": True}, cfg.profiles, "flood")
    ref = json.loads((OUT / "fp16_ref.json").read_text()) if name != "fp16_ref" and (OUT / "fp16_ref.json").exists() else None
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for i, (idx, t, frame, actors) in enumerate(frames):
        torch.cuda.synchronize()
        a = an.analyze(frame, actors, force=True)
        torch.cuda.synchronize()
        row = {"frame": idx, "t": round(t, 2), "latency_s": round(a.latency_s, 2),
               "decoder": a.timings.get("decoder"), "tok_s": a.timings.get("decode_tok_per_s"),
               "new_tokens": (a.timings.get("turn1_new_tokens", 0) + a.timings.get("turn2_new_tokens", 0)),
               "scene": a.scene, "alert_level": a.alert_level, "hazards": a.hazards,
               "actors": a.actors, "n_actor_boxes": len(actors), "raw_report": a.timings.get("raw_report")}
        if ref is not None:
            row["tf_agree"], row["tf_nll"] = teacher_forced(an, model, frame, actors, ref["rows"][i]["scene"])
        rows.append(row)
        print(f"[{tag}] frame {idx}: {a.latency_s:.2f}s [{a.alert_level}] {a.scene[:100]}"
              + (f"  agree={row['tf_agree']:.3f} nll={row['tf_nll']:.3f}" if "tf_agree" in row else ""))
    negs = []
    for pth, img in neg_frames():
        a = an.analyze(img, [], force=True)
        negs.append({"image": pth, "alert_level": a.alert_level, "scene": a.scene,
                     "hazards": a.hazards})
        print(f"[{tag}] negative {Path(pth).name}: [{a.alert_level}] {a.scene[:100]}")
    res = {"variant": name, "tag": tag, "negatives": negs, "adapter": adapter, "desc": VARIANTS.get(name, ""),
           "load_s": round(load_s, 1), "load_vram_gb": round(load_gb, 2),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "offloaded_to": offloaded, "fast_decode": an.fast_decode_active, "rows": rows}
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[{tag}] saved {OUT / (tag + '.json')}")


def task_score(rows, labels):
    good_alert = sum(r["alert_level"] in labels["alert_ok"] for r in rows)
    flood = sum(any(h["type"] in labels["flood_types"] for h in r["hazards"]) for r in rows)
    fast = sum(any(h["type"] in labels["fast_types"] for h in r["hazards"]) or
               any(w in r["scene"].lower() for w in ("rapid", "fast", "swift", "strong current", "torrent"))
               for r in rows)
    wrong_calm = sum(any(p in r["scene"].lower() for p in labels["contradictions"]) for r in rows)
    parsed = sum(r["alert_level"] != "unknown" for r in rows)
    n = len(rows)
    return {"alert_ok": good_alert / n, "flood_reported": flood / n, "fast_water": fast / n,
            "says_calm": wrong_calm / n, "json_ok": parsed / n}


def report() -> None:
    labels = json.loads((APP_DIR / "benchmarks" / "quant_labels.json").read_text())
    files = sorted(OUT.glob("*.json"))
    rows = []
    for f in files:
        r = json.loads(f.read_text())
        rr = r["rows"]
        sc = task_score([x for x in rr if x["frame"] >= HELDOUT_FROM], labels)
        sc_all = task_score(rr, labels)
        negs = r.get("negatives", [])
        sc["neg_false_alarm"] = (sum(n["alert_level"] not in ("none", "low") for n in negs) / len(negs)) if negs else None
        sc["alert_all"] = sc_all["alert_ok"]
        lat = np.mean([x["latency_s"] for x in rr])
        tps = [x["tok_s"] for x in rr if x.get("tok_s")]
        agree = [x["tf_agree"] for x in rr if "tf_agree" in x]
        nll = [x["tf_nll"] for x in rr if "tf_nll" in x]
        rows.append((r["tag"], r["load_vram_gb"], r["peak_vram_gb"], lat, np.mean(tps) if tps else None,
                     np.mean(agree) if agree else None, np.mean(nll) if nll else None, sc, r["offloaded_to"]))
    print("accuracy columns = held-out frames >= %d (frames the QLoRA adapter never saw)" % HELDOUT_FROM)
    print(f"{'variant':<22}{'VRAM':>6}{'peak':>6}{'lat s':>7}{'tok/s':>7}{'agree':>7}{'NLL':>7}"
          f"{'alert':>7}{'flood':>7}{'fast':>6}{'calm!':>7}{'json':>6}{'allfr':>7}{'negFA':>7}")
    for tag, lv, pv, lat, tps, ag, nl, sc, off in rows:
        f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
        print(f"{tag:<22}{lv:>6.2f}{pv:>6.2f}{lat:>7.2f}{f(tps, 1):>7}{f(ag, 3):>7}{f(nl, 3):>7}"
              f"{sc['alert_ok']:>7.0%}{sc['flood_reported']:>7.0%}{sc['fast_water']:>6.0%}"
              f"{sc['says_calm']:>7.0%}{sc['json_ok']:>6.0%}{sc['alert_all']:>7.0%}"
              f"{('-' if sc['neg_false_alarm'] is None else format(sc['neg_false_alarm'], '.0%')):>7}"
              + (f"  (offload: {off})" if off else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter to apply")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run_variant(a.variant, a.adapter, a.tag or a.variant)
