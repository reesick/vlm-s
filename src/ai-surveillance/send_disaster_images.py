"""Describe each disaster image with the VLM and send it to the Bitchat phone.

For every image in data/disaster_images (sorted by name):
  1. Qwen2.5-VL-3B (NF4 double-quant, base model, no flood adapter) analyses it
     through the same fast path as the live pipeline (vlm/analyzer.py):
     256/512-token image budget, CUDA-graph decoder, and turn 2 (alert level +
     hazards JSON) continues turn 1's KV cache, so the image is encoded once.
     Turn 1 is a detailed paragraph like the PPT screenshots (slide 7).
     Every message text comes from the VLM; if the VLM fails, nothing is sent.
  2. POST /send/image  (the photo, no caption: the description follows as text)
  3. wait for the phone's rate window (rate_limit_ms from /status, +1 s)
  4. POST /send/text   (the description)
  5. wait again before the next image.

  python send_disaster_images.py --ip 192.168.1.5
  python send_disaster_images.py --ip 192.168.1.5 --dry-run   # describe only, send nothing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import requests

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

PROMPT = (
    "Describe this image in one detailed paragraph of 80 to 120 words for an emergency "
    "responder who cannot see it. Say what kind of disaster or emergency it shows, what "
    "is damaged, what the people are doing, any visible dangers, and the overall situation. "
    "Be factual and only describe what is visible."
)


def jpeg_for_mesh(img, max_dim=640, quality=70) -> bytes:
    h, w = img.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def split_text(text: str, max_part: int) -> list[str]:
    """Split on sentence/word boundaries into parts of at most max_part chars."""
    import textwrap
    out = []
    for para in [p for p in text.split("\n") if p.strip()]:
        out += textwrap.wrap(para, max_part) or []
    return out or [text[:max_part]]


def post(url, retries=3, **kw):
    for k in range(retries):
        try:
            r = requests.post(url, timeout=15, **kw)
            try:
                body = r.json()
            except ValueError:
                body = r.text[:120]
            return r.status_code, body
        except requests.RequestException as e:
            print(f"   attempt {k + 1} failed: {e}")
            time.sleep(2)
    return None, "failed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dir", default="data/disaster_images")
    ap.add_argument("--channel", default=None, help="Bitchat channel; default = public chat")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-part", type=int, default=190,
                    help="max chars per text message (app mesh packets fragment above 512 bytes)")
    a = ap.parse_args()

    base = f"http://{a.ip}:{a.port}"
    gap = 6.0
    if not a.dry_run:
        try:
            st = requests.get(base + "/status", timeout=6).json()
        except Exception as e:  # noqa: BLE001
            print(f"[send] phone not reachable at {base}: {e}")
            return 2
        if not st.get("api_enabled"):
            print("[send] Bitchat VLM API is disabled on the phone (Settings -> VLM API)")
            return 2
        gap = st.get("rate_limit_ms", 5000) / 1000.0 + 1.0
        print(f"[send] phone OK  mesh={st.get('mesh_running')}  peers={st.get('peers_count')}  gap={gap:.1f}s")

    files = sorted(p for p in (APP_DIR / a.dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        print(f"[send] no images in {a.dir}")
        return 1

    # Base model on purpose: the flood QLoRA adapter makes every scene sound like a flood.
    from vlm.core import VLMConfig, VLMCore
    cfg = VLMConfig.from_yaml()
    cfg.lora_adapter = ""
    cfg.analyzer_cfg = {**cfg.analyzer_cfg, "warmup": True, "skip_if_unchanged": False,
                        "scene_max_new_tokens": 200, "mark_actors": False}
    cfg.profiles = {**cfg.profiles, "report": {**cfg.profiles.get("generic", {}),
                                              "description": "disaster photo report",
                                              "scene_prompt": PROMPT,
                                              "hazard_types": ["building_collapse", "fire", "smoke", "wildfire",
                                                               "flood", "landslide", "storm_damage", "debris",
                                                               "trapped_people", "injured_people",
                                                               "rescue_activity", "other"],
                                              "actor_states": ["rescuing", "trapped", "injured", "evacuating",
                                                               "wading", "observing", "unknown"]}}
    core = VLMCore(cfg)
    if not core.initialize():
        print("[send] VLM failed to load; nothing sent")
        return 3
    an = core.analyzer
    an.set_profile("report")

    results = []
    last_send = 0.0
    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        if img is None:
            print(f"[send] skip unreadable {f.name}")
            continue
        res = an.analyze(img, actors=None, force=True, max_pixels=401408)
        tm = res.timings
        if not res.scene.strip():
            raise RuntimeError(f"[send] VLM returned an empty description for {f.name}; stopping")
        hz = ", ".join(h["type"] for h in res.hazards) or "none"
        text = f"{res.scene.strip()}\n\nAlert: {res.alert_level.upper()} | Hazards: {hz}"
        print(f"   vlm: {tm.get('decoder')}  kv_reused={tm.get('kv_reused')}  "
              f"image_tokens={tm.get('image_tokens')}  {tm.get('decode_tok_per_s')} tok/s")
        dt = res.latency_s
        print(f"\n[{i}/{len(files)}] {f.name}  ({dt:.1f}s)\n{text}")
        rec = {"file": f.name, "description": res.scene, "alert_level": res.alert_level,
               "hazards": res.hazards, "message": text, "vlm_s": round(dt, 2),
               "kv_reused": tm.get("kv_reused"), "decoder": tm.get("decoder"),
               "image_tokens": tm.get("image_tokens")}

        if not a.dry_run:
            jpg = jpeg_for_mesh(img)
            wait = gap - (time.perf_counter() - last_send)
            if wait > 0:
                time.sleep(wait)
            form = {}
            if a.channel:
                form["channel"] = a.channel
            code, body = post(base + "/send/image", files={"image": (f.name, jpg, "image/jpeg")}, data=form)
            last_send = time.perf_counter()
            print(f"   image {len(jpg) // 1024} KB -> {code} {body}")
            rec["image_status"] = code
            time.sleep(gap)
            # The app wraps each text in an IDX1 camera packet (~300 bytes) and mesh
            # messages over 512 bytes are fragmented and do not arrive. Keep every
            # part <= MAX_PART chars so text + wrapper stays under 512 bytes.
            parts = split_text(text, a.max_part)
            rec["text_status"] = []
            for k, part in enumerate(parts, 1):
                if k > 1:
                    time.sleep(gap)
                t = f"({k}/{len(parts)}) {part}" if len(parts) > 1 else part
                payload = {"text": t}
                if a.channel:
                    payload["channel"] = a.channel
                code, body = post(base + "/send/text", json=payload)
                if isinstance(body, dict) and "Rate limited" in str(body.get("error", "")):
                    time.sleep(gap)
                    code, body = post(base + "/send/text", json=payload)
                last_send = time.perf_counter()
                print(f"   text {k}/{len(parts)} ({len(t)} chars) -> {code} {body}")
                rec["text_status"].append(code)
        results.append(rec)

    out = APP_DIR / a.dir / "descriptions.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[send] done: {len(results)} images. Descriptions saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
