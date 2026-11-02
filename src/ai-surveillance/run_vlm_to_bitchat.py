"""Live camera -> VLM description on screen -> same frame + description to Bitchat.

  python run_vlm_to_bitchat.py                    # A36 smart camera (index 2, DSHOW)
  python run_vlm_to_bitchat.py --camera 0         # laptop webcam
  python run_vlm_to_bitchat.py --no-share         # describe on screen only
  python run_vlm_to_bitchat.py --api https://<command-centre> --key <MESH_GATEWAY_KEY> --lat 18.65 --lon 73.76
                                                  # also post every frame + description to the command centre
  python run_vlm_to_bitchat.py --api ... --key ... --demo-pin
                                                  # camera placed inside the PCCOE drone map, labelled demo position

Keys in the video window:
  S      toggle sharing to Bitchat on/off (shown top-right)
  SPACE  describe the current frame now
  Q/Esc  quit

How it decides when to describe: every --every seconds, but a frame that looks
the same as the last described one (64x36 thumbnail diff) is skipped, so the VLM
and the mesh are only used when the scene changes. FIRE/FLOOD-like alerts are
shared immediately; repeated identical descriptions are not re-sent.

Sharing = public Bitchat broadcast: every participant on the mesh receives the
frame and its description (visible to all roles, "All" chip).
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import textwrap
import threading
import time
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

from core.command_centre import DEMO_PIN, CommandCentreLink  # noqa: E402
from core.mesh_alert import BitchatSender, MeshAlert  # noqa: E402

PROMPT = (
    "You are a live safety camera. In 2 to 3 sentences (under 60 words) describe what is "
    "happening right now: the place, the people and what they are doing, and any danger "
    "such as fire, smoke, flood, a fall, a fight or an injury. If everything is normal, say so."
)
HAZARDS = ["fire", "smoke", "flood", "fall", "fight", "injured_person", "weapon", "crowd",
           "building_collapse", "accident", "other"]


def open_camera(index: int, wait_s: float):
    t0, last = time.time(), -10.0
    while time.time() - t0 < wait_s:
        for be in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            cap = cv2.VideoCapture(index, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                t1 = time.time()
                while time.time() - t1 < 3:
                    ok, f = cap.read()
                    if ok and f is not None and f.mean() > 5:
                        print(f"[cam] camera {index} live {f.shape[1]}x{f.shape[0]} after {time.time() - t0:.0f}s")
                        return cap
            cap.release()
        if time.time() - last >= 10:
            last = time.time()
            print(f"[cam] waiting for camera {index} ... {time.time() - t0:.0f}s (open the camera app on the phone)")
        time.sleep(1)
    raise RuntimeError(f"camera {index} sent no frames in {wait_s:.0f}s")


def thumb(img):
    return cv2.cvtColor(cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)


def draw(frame, state, share_on, fps):
    vis = frame.copy()
    h, w = vis.shape[:2]
    status = "VLM thinking..." if state["busy"] else f"VLM {state['latency']:.1f}s"
    share = "SHARING TO BITCHAT" if share_on else "sharing OFF"
    cv2.rectangle(vis, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(vis, f"fps {fps:4.1f} | {status} | sent {state['sent']}", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(share, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(vis, share, (w - tw - 10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 220, 0) if share_on else (0, 0, 255), 2, cv2.LINE_AA)
    text = state["text"]
    if text:
        scale = max(0.55, w / 1600)
        cw = cv2.getTextSize("M", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] * 0.8
        lines = textwrap.wrap(text, max(20, int((w - 20) / cw)))[:6]
        lh = int(30 * scale) + 6
        y0 = h - lh * len(lines) - 12
        ov = vis.copy()
        cv2.rectangle(ov, (0, y0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.65, vis, 0.35, 0, vis)
        col = (0, 0, 255) if state["level"] in ("high", "critical") else (0, 255, 180)
        for i, ln in enumerate(lines):
            cv2.putText(vis, ln, (10, y0 + lh * (i + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
    return vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=2, help="2 = Samsung A36 smart camera, 0 = laptop webcam")
    ap.add_argument("--wait", type=float, default=180, help="seconds to wait for the camera to start")
    ap.add_argument("--every", type=float, default=8.0, help="seconds between VLM looks")
    ap.add_argument("--change", type=float, default=6.0, help="min scene change to describe again")
    ap.add_argument("--ip", default="", help="phone IP (empty = find it on this Wi-Fi)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-share", action="store_true")
    ap.add_argument("--no-command-centre", action="store_true",
                    help="do not send the extra IDX1|S copy the gateway forwards to the command centre")
    ap.add_argument("--cc-url", default="", help="command-centre API base URL to set on the phone's gateway")
    ap.add_argument("--cc-key", default="", help="gateway key for the command centre")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until Q)")
    ap.add_argument("--api", default=os.environ.get("INDRADHANU_API", ""),
                    help="command-centre base URL; every described frame is POSTed to /api/v1/ingest/camera")
    ap.add_argument("--key", default=os.environ.get("MESH_GATEWAY_KEY", ""), help="X-Mesh-Gateway-Key for --api")
    ap.add_argument("--lat", type=float, default=None, help="where this camera is (frames have no GPS)")
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--demo-pin", action="store_true",
                    help=f"place the camera inside the PCCOE drone map ({DEMO_PIN[0]}, {DEMO_PIN[1]}) and label it a demo position")
    ap.add_argument("--node", default=os.environ.get("VLM_NODE_ID", "vlm-cam-01"), help="camera id shown in the command centre")
    a = ap.parse_args()

    cc = None
    api = a.api or a.cc_url
    if api:
        key = a.key or a.cc_key
        if a.demo_pin or a.lat is None or a.lon is None:
            if not a.demo_pin:
                print("[cc] no --lat/--lon given: using the demo position inside the PCCOE drone map")
            lat, lon, demo = DEMO_PIN[0], DEMO_PIN[1], True
        else:
            lat, lon, demo = a.lat, a.lon, False
        if not key:
            print("[cc] --api given without --key (MESH_GATEWAY_KEY): the command centre will refuse the frames")
        cc = CommandCentreLink(api, key, lat=lat, lon=lon, node_id=a.node, demo_position=demo)
        threading.Thread(target=cc.check, daemon=True).start()
        # The direct post already files hazards; the phone's IDX1|S copy would file them a second time
        # from the phone's own location. The mesh broadcast to phones is unchanged.
        a.no_command_centre = True

    share = {"on": not a.no_share}
    sender = None
    if share["on"]:
        try:
            sender = BitchatSender.connect(a.ip, a.port)
        except ConnectionError as e:
            print(f"[mesh] {e}\n[mesh] continuing with on-screen descriptions only")
            share["on"] = False
    if sender is not None:
        if a.cc_url:
            print(f"[cc] configuring phone gateway -> {a.cc_url}: {sender.configure_gateway(a.cc_url, a.cc_key)}")
        g = sender.gateway_status()
        print(f"[cc] gateway enabled={g.get('enabled')} url={g.get('api_url') or '-'} key_set={g.get('gateway_key_set')} "
              f"online={g.get('online')} queued={g.get('queued')} last={g.get('last_result')}")
        if not g.get("enabled") and not a.no_command_centre:
            # Gateway off: the extra command-centre copy would only cost a rate window
            # per alert (fewer live frames reach the mesh) and go nowhere.
            print("[cc] gateway is off on the phone -> skipping the command-centre copy")
            a.no_command_centre = True

    from vlm.core import VLMConfig, VLMCore
    cfg = VLMConfig.from_yaml()
    cfg.lora_adapter = ""
    cfg.analyzer_cfg = {**cfg.analyzer_cfg, "warmup": True, "skip_if_unchanged": False,
                        "scene_max_new_tokens": 90, "mark_actors": False}
    cfg.profiles = {**cfg.profiles, "live": {**cfg.profiles.get("generic", {}),
                                            "description": "live safety camera",
                                            "scene_prompt": PROMPT, "hazard_types": HAZARDS}}
    core = VLMCore(cfg)
    if not core.initialize():
        print("[vlm] failed to load")
        return 3
    an = core.analyzer
    an.set_profile("live")

    cap = open_camera(a.camera, a.wait)

    state = {"text": "Waiting for the first description...", "level": "", "busy": False,
             "latency": 0.0, "sent": 0}
    lock = threading.Lock()
    outbox: queue.Queue = queue.Queue(maxsize=1)      # newest alert only
    stop = threading.Event()

    def sender_loop():
        while not stop.is_set():
            try:
                alert = outbox.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if sender.send(alert, command_centre=not a.no_command_centre).ok:
                    with lock:
                        state["sent"] += 1
            except Exception as e:  # noqa: BLE001
                print(f"[mesh] send failed: {e}")

    if sender is not None:
        threading.Thread(target=sender_loop, daemon=True).start()

    last_desc = {"t": -1e9, "thumb": None, "text": ""}

    def describe(frame):
        try:
            res = an.analyze(frame, actors=None, force=True)
            text = res.scene.strip()
            hz = [h["type"] for h in res.hazards]
            with lock:
                state.update(text=f"[{res.alert_level.upper()}] {text}", level=res.alert_level,
                             latency=res.latency_s)
            print(f"[vlm] {res.latency_s:.1f}s [{res.alert_level}] {text}  hazards={hz}")
            if cc is not None and text and text != last_desc["text"]:
                cc.offer(frame, text, res.alert_level, hz)
            if share["on"] and sender is not None and text and text != last_desc["text"]:
                alert = MeshAlert(image=frame, text=text, alert_level=res.alert_level,
                                  hazards=hz, source=f"cam{a.camera}_{time.strftime('%H%M%S')}.jpg")
                try:
                    outbox.get_nowait()          # drop an older unsent alert
                except queue.Empty:
                    pass
                outbox.put_nowait(alert)
            last_desc["text"] = text
        except Exception as e:  # noqa: BLE001
            print(f"[vlm] error: {e}")
        finally:
            with lock:
                state["busy"] = False

    win = "VLM live -> Bitchat"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t_start, fps, n, t_fps = time.time(), 0.0, 0, time.time()
    force = True
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        n += 1
        if time.time() - t_fps >= 1.0:
            fps, n, t_fps = n / (time.time() - t_fps), 0, time.time()

        with lock:
            busy = state["busy"]
        now = time.time()
        if not busy and (force or now - last_desc["t"] >= a.every):
            th = thumb(frame)
            changed = last_desc["thumb"] is None or float(np.mean(np.abs(th - last_desc["thumb"]))) >= a.change
            if force or changed:
                last_desc.update(t=now, thumb=th)
                with lock:
                    state["busy"] = True
                threading.Thread(target=describe, args=(frame.copy(),), daemon=True).start()
            else:
                last_desc["t"] = now
            force = False

        cv2.imshow(win, draw(frame, state, share["on"] and sender is not None, fps))
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord(" "):
            force = True
        if k == ord("s") and sender is not None:
            share["on"] = not share["on"]
            print(f"[mesh] sharing {'ON' if share['on'] else 'OFF'}")
        if a.seconds and now - t_start > a.seconds:
            break

    stop.set()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] alerts sent: {state['sent']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
