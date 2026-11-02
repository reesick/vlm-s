"""Describe images with the VLM and send each as one MeshAlert (image + its text).

  python send_vlm_alerts.py                         # all images in data/disaster_images
  python send_vlm_alerts.py --ip 172.20.10.4        # preferred phone IP (auto-found if moved)
  python send_vlm_alerts.py --dir path\\to\\images --short

Every description comes from the VLM; if the VLM fails on an image, nothing is
sent for that image.
"""
from __future__ import annotations

import argparse
import os
import time
import re
import sys
from pathlib import Path

import cv2

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

from core.mesh_alert import BitchatSender, MeshAlert  # noqa: E402

PROMPT = (
    "Describe this image in one detailed paragraph of 60 to 100 words for an emergency "
    "responder who cannot see it. Say what kind of disaster or emergency it shows, what "
    "is damaged, what the people are doing, any visible dangers, and the overall situation. "
    "Be factual and only describe what is visible."
)
HAZARDS = ["building_collapse", "fire", "smoke", "wildfire", "flood", "landslide", "storm_damage",
           "debris", "trapped_people", "injured_people", "rescue_activity", "other"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="", help="phone IP (empty = find it on this Wi-Fi)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dir", default="data/disaster_images")
    ap.add_argument("--short", action="store_true", help="send only the first sentence of each description")
    ap.add_argument("--api", default=os.environ.get("INDRADHANU_API", ""), help="command-centre base URL (also POST each image there)")
    ap.add_argument("--key", default=os.environ.get("MESH_GATEWAY_KEY", ""), help="X-Mesh-Gateway-Key for --api")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--demo-pin", action="store_true", help="place the camera inside the PCCOE drone map (demo position)")
    ap.add_argument("--node", default=os.environ.get("VLM_NODE_ID", "vlm-cam-01"))
    ap.add_argument("--gap", type=float, default=20.0, help="seconds between images when posting to the command centre")
    a = ap.parse_args()

    files = sorted(p for p in (APP / a.dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        print(f"no images in {a.dir}")
        return 1
    cc = None
    if a.api:
        from core.command_centre import DEMO_PIN, CommandCentreLink, Frame
        demo = a.demo_pin or a.lat is None or a.lon is None
        cc = CommandCentreLink(a.api, a.key, lat=DEMO_PIN[0] if demo else a.lat, lon=DEMO_PIN[1] if demo else a.lon,
                               node_id=a.node, demo_position=demo)
        cc.check()
    try:
        sender = BitchatSender.connect(a.ip, a.port)
    except ConnectionError as e:
        if cc is None:
            raise
        print(f"[mesh] {e}\n[mesh] no phone: images go to the command centre only")
        sender = None

    from vlm.core import VLMConfig, VLMCore
    cfg = VLMConfig.from_yaml()
    cfg.lora_adapter = ""          # base model: the flood adapter makes every scene a flood
    cfg.analyzer_cfg = {**cfg.analyzer_cfg, "warmup": True, "skip_if_unchanged": False,
                        "scene_max_new_tokens": 160, "mark_actors": False}
    cfg.profiles = {**cfg.profiles, "report": {**cfg.profiles.get("generic", {}),
                                              "description": "disaster photo report",
                                              "scene_prompt": PROMPT, "hazard_types": HAZARDS}}
    core = VLMCore(cfg)
    if not core.initialize():
        print("VLM failed to load; nothing sent")
        return 3
    an = core.analyzer
    an.set_profile("report")

    sent = 0
    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        if img is None:
            continue
        res = an.analyze(img, actors=None, force=True, max_pixels=401408)
        desc = res.scene.strip()
        if not desc:
            print(f"[{i}/{len(files)}] {f.name}: VLM gave no description, skipped")
            continue
        if a.short:
            desc = re.split(r"(?<=\.)\s", desc)[0]
        alert = MeshAlert(image=img, text=desc, alert_level=res.alert_level,
                          hazards=[h["type"] for h in res.hazards], source=f.name)
        print(f"\n[{i}/{len(files)}] {f.name}  VLM {res.latency_s:.1f}s  "
              f"({res.timings.get('decoder')}, kv_reused={res.timings.get('kv_reused')})\n  {alert.headline()}")
        if cc is not None:
            cc._post(Frame(
                image=img, text=desc, alert_level=res.alert_level, hazards=alert.hazards, at=time.time()))
        if sender is not None and sender.send(alert, command_centre=cc is None).ok:
            sent += 1
        elif sender is None and cc is not None:
            sent += 1
            time.sleep(a.gap)
    print(f"\ndone: {sent}/{len(files)} alerts delivered to {sender.base if sender else a.api}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
