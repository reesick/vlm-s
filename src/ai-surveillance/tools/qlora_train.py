"""QLoRA fine-tune of the deployed Qwen2.5-VL-3B (NF4 double-quant base) for the analyzer task.

The base model (fp16 or 4-bit alike) calls fast floodwater "moderate, slow flow"
(benchmarks/quant_eval.py: 0% correct alert level even at fp16), so the gap is
task knowledge, not quantization. QLoRA trains small LoRA adapters on the
frozen 4-bit model to fix that without changing the deployment format: the
adapter is merged into the NF4 weights at load time (vlm/core.py
model.lora_adapter), so speed and VRAM stay those of the base model.

Training data = exactly the token sequence the analyzer produces at inference
(turn 1 scene prompt + image with Set-of-Mark ids, turn 2 report prompt with the
JSON prefill), with hand-written targets:
  * positives: frames 0-169 of the flood clip (frames 180+ are held out for eval)
  * negatives: indoor camera frames with no water (alert none), so the adapter
    does not learn "always say flood"
Loss only on the assistant tokens.

  python tools/qlora_train.py --out models/lora/flood_v1 [--epochs 3] [--rank 16]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

VIDEO = APP_DIR.parents[1] / "WhatsApp Video 2026-10-10 at 11.34.09 AM.mp4"
TRAIN_MAX_FRAME = 169          # frames 180-324 are the held-out eval split
NEGATIVES = {                  # indoor camera frames, no water
    "data/keyframes/FALL_2026-10-30T12-10-03_f363_id1.jpg":
        "An indoor room where a young man in a white shirt sits close to the camera; there is no water or flooding anywhere.",
    "data/keyframes/FALL_2026-10-30T12-10-39_f999_id1.jpg":
        "An indoor room with two young men close to the camera, one reaching toward it; the floor is dry and no water is visible.",
}
POS_SCENES = [
    "Fast, turbulent brown floodwater surges on both sides of a long, damaged bridge, and {who} crossing the bridge on foot directly above the raging current.",
    "Violent muddy floodwater is rushing past a narrow bridge whose deck is partly washed over, while {who} crossing it on foot in serious danger.",
    "A powerful flood of fast-flowing brown water churns around a long bridge; {who} on the bridge, walking across above the dangerous current.",
]


def who_text(n: int) -> str:
    if n <= 0:
        return "a few people are"
    if n == 1:
        return "one person is"
    if n == 2:
        return "two people are"
    return f"{n} people are" if n < 10 else "many people are"


class _T:
    def __init__(self, tid, cls, conf, xyxy):
        self.track_id, self.cls, self.conf, self.xyxy = tid, cls, conf, xyxy


class _FixedMotion:
    def __init__(self, m):
        self.m = m

    def describe(self, tid):
        return self.m.get(tid, "unknown")


def track_video():
    """Continuous YOLO+ByteTrack over the clip; per frame: tracks + motion text."""
    from ultralytics import YOLO
    from vlm.analyzer import TrackMotionHistory
    yolo = YOLO(str(APP_DIR / "models" / "yolov8n.pt"))
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hist = TrackMotionHistory()
    frames, per = [], []
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        r = yolo.track(fr, persist=True, verbose=False, classes=[0, 1, 2, 3, 5, 7, 8],
                       conf=0.3, tracker="bytetrack.yaml")[0]
        tr = []
        if r.boxes is not None and r.boxes.id is not None:
            for b, tid, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                     r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                tr.append(_T(tid, c, cf, tuple(int(v) for v in b)))
        hist.update(tr, t=fi / fps)
        frames.append(fr)
        per.append((tr, {t.track_id: hist.describe(t.track_id) for t in tr}))
        fi += 1
    cap.release()
    del yolo
    return frames, per


def augment(frame, tracks, rng):
    """Random crop (85-100%) + brightness/contrast jitter; boxes follow the crop."""
    h, w = frame.shape[:2]
    s = rng.uniform(0.85, 1.0)
    cw, ch = int(w * s), int(h * s)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    img = frame[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    img = np.clip(img * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15), 0, 255).astype(np.uint8)
    out = []
    for t in tracks:
        x1, y1, x2, y2 = t.xyxy
        x1, y1, x2, y2 = max(0, x1 - x0), max(0, y1 - y0), min(cw - 1, x2 - x0), min(ch - 1, y2 - y0)
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            out.append(_T(t.track_id, t.cls, t.conf, (x1, y1, x2, y2)))
    return img, out


def targets_positive(actors, rng):
    persons = [a for a in actors if a.kind == "person"]
    n = len(persons) if actors else 2
    scene = rng.choice(POS_SCENES).format(who=who_text(n))
    hazards = [{"type": "flood", "confidence": 0.95, "description": "turbulent muddy floodwater"},
               {"type": "fast_current", "confidence": 0.9, "description": "fast surging current"},
               {"type": "person_crossing", "confidence": 0.85, "description": "people crossing bridge"},
               {"type": "bridge_damaged", "confidence": 0.6, "description": "deck washed with mud"}]
    if actors:
        ents = [{"id": ",".join(a.id for a in persons), "state": "on_bridge",
                 "action": "crossing bridge", "risk": "high"}] if persons else []
        for a in actors:
            if a.kind != "person":
                ents.append({"id": a.id, "state": "unknown", "action": "near floodwater", "risk": "medium"})
    else:
        ents = [{"id": "A1,A2", "kind": "person", "state": "on_bridge", "action": "crossing bridge", "risk": "high"}]
    return scene, {"alert_level": "high", "hazards": hazards, "actors": ents}


def targets_negative(scene, actors):
    ents = [{"id": ",".join(a.id for a in actors if a.kind == "person"), "state": "safe_ground",
             "action": "indoors", "risk": "none"}] if any(a.kind == "person" for a in actors) else []
    return scene, {"alert_level": "none", "hazards": [], "actors": ents}


def build_sample(an, frame, actors, scene_t, report_t, extra, device):
    """Token ids + labels for the full two-turn conversation, as analyze() runs it."""
    import torch
    from vlm.analyzer import budget_size, draw_actor_marks, resize_for_budget
    budget = int(an.cfg.get("image_max_pixels", 200704))
    h, w = frame.shape[:2]
    nw, _ = budget_size(w, h, budget)
    marked = bool(actors)
    src = draw_actor_marks(frame, actors, nw / float(w)) if marked else frame
    rgb = cv2.cvtColor(resize_for_budget(src, budget), cv2.COLOR_BGR2RGB)
    inputs = an._encode(an._scene_prompt(actors, marked), rgb)
    tok = an.tokenizer
    p1 = inputs["input_ids"]
    a1 = tok(scene_t + "<|im_end|>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    p2 = an._turn2_suffix_ids(actors, extra, device)       # ends with the JSON prefill
    js = json.dumps(report_t, separators=(", ", ": "))
    prefill = an._json_prefill()
    assert js.startswith(prefill), (js[:30], prefill)
    a2 = tok(js[len(prefill):] + "<|im_end|>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full = torch.cat([p1, a1, p2, a2], dim=1)
    labels = full.clone()
    L1, A1, P2 = p1.shape[1], a1.shape[1], p2.shape[1]
    labels[:, :L1] = -100
    labels[:, L1 + A1:L1 + A1 + P2] = -100
    fresh = an._fresh_inputs(inputs, full)
    fresh["labels"] = labels
    return fresh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/lora/flood_v1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--stride", type=int, default=3, help="use every Nth train frame")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("[qlora] tracking the clip ...")
    frames, per = track_video()
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from vlm.analyzer import VLMAnalyzer, actors_from_tracks
    from vlm.core import VLMConfig
    cfg = VLMConfig.from_yaml()

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    # bf16 for training: fp16 overflowed to NaN in epoch 2 of the first run.
    # Inference still runs fp16 compute; the merged LoRA delta is dtype-agnostic.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_name, quantization_config=q, torch_dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True)
    proc = AutoProcessor.from_pretrained(cfg.model_name)
    an = VLMAnalyzer(None, proc, {**cfg.analyzer_cfg, "fast_decode": False, "warmup": False},
                     cfg.profiles, "flood")
    an.model = model
    dev = torch.device("cuda:0")

    # ---- dataset (built once; images augmented per epoch) ----
    items = []
    for fi in range(0, min(TRAIN_MAX_FRAME + 1, len(frames)), args.stride):
        items.append(("pos", frames[fi], per[fi][0], per[fi][1], None))
    for path, text in NEGATIVES.items():
        img = cv2.imread(str(APP_DIR / path))
        if img is None:
            print(f"[qlora] WARN missing negative {path}")
            continue
        for _ in range(6):     # repeated with different augmentations
            items.append(("neg", img, [], {}, text))
    print(f"[qlora] samples/epoch: {len(items)}  ({sum(i[0] == 'pos' for i in items)} pos, "
          f"{sum(i[0] == 'neg' for i in items)} neg)")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    an.model = model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = math.ceil(len(items) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 5) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))

    model.train()
    step, t0, log = 0, time.perf_counter(), []
    torch.cuda.reset_peak_memory_stats()
    for ep in range(args.epochs):
        rng.shuffle(items)
        run_loss, n = 0.0, 0
        for k, (kind, img, tracks, motion, neg_text) in enumerate(items):
            aimg, atr = augment(img, tracks, rng)
            actors = actors_from_tracks(atr, _FixedMotion(motion), aimg.shape)
            if kind == "pos":
                scene_t, rep_t = targets_positive(actors, rng)
                extra = (f"Water sensor: {rng.randint(78, 90)}% of view is water, "
                         f"{rng.randint(40, 70)}% of it moving.") if rng.random() < 0.5 else ""
            else:
                scene_t, rep_t = targets_negative(neg_text, actors)
                extra = ""
            batch = build_sample(an, aimg, actors, scene_t, rep_t, extra, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / args.accum
            if not torch.isfinite(loss):
                print(f"[qlora] WARN non-finite loss at epoch {ep + 1} item {k}; skipped")
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            run_loss += float(loss) * args.accum
            n += 1
            if (k + 1) % args.accum == 0 or k == len(items) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
        log.append({"epoch": ep + 1, "loss": run_loss / max(1, n)})
        print(f"[qlora] epoch {ep + 1}/{args.epochs} loss={run_loss / max(1, n):.4f} "
              f"steps={step} elapsed={time.perf_counter() - t0:.0f}s "
              f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GB")

    out = APP_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    (out / "train_log.json").write_text(json.dumps({
        "args": vars(args), "log": log, "samples": len(items), "train_frames": f"0-{TRAIN_MAX_FRAME}",
        "train_s": round(time.perf_counter() - t0, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}, indent=2))
    print(f"[qlora] adapter saved to {out}")


if __name__ == "__main__":
    main()
