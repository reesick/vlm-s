"""Send each disaster image followed by its VLM description, readable by every role.

Why texts were invisible: the phone app (VlmMessageHandler.sendTextMessage) wraps
every API text as a camera packet "text IDX1|S|{...}|-", and AudiencePolicy hides
IDX1|S| packets on any phone whose role is not Gov/Command. If the text already
contains "IDX1|", the app sends it unchanged, so we append a neutral packet
" IDX1|N|{}|-": not a sensor packet (not hidden), ignored by the gateway, and
stripped from the chat bubble by stripMachinePacket(). category=UNSET keeps the
app from filing it under Fire/Gov chips. Parts stay <=180 chars (512-byte mesh
packet limit).

Look at the OTHER phone (the peer), with the "All" chip selected.

  python send_short.py --ip 172.20.10.4            # full description in parts
  python send_short.py --ip 172.20.10.4 --short    # one short line per image
"""
import argparse
import json
import re
import sys
import textwrap
import time
from pathlib import Path

import cv2
import requests

APP = Path(__file__).resolve().parent
TAG = " IDX1|N|{}|-"

ap = argparse.ArgumentParser()
ap.add_argument("--ip", default="172.20.10.4")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--short", action="store_true", help="send only the first sentence")
ap.add_argument("--part", type=int, default=180)
a = ap.parse_args()


def find_phone(port):
    """Scan the laptop's /24 networks for a Bitchat API (like send_to_phone.ps1)."""
    import socket
    from concurrent.futures import ThreadPoolExecutor
    nets = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith(("127.", "169.254.")):
                nets.add(ip.rsplit(".", 1)[0])
    except OSError:
        pass

    def probe(ip):
        try:
            with socket.create_connection((ip, port), timeout=0.4):
                pass
            return ip if requests.get(f"http://{ip}:{port}/status", timeout=3).json().get("api_enabled") is not None else None
        except Exception:
            return None

    for net in sorted(nets):
        print(f"scanning {net}.1-254 for port {port} ...")
        with ThreadPoolExecutor(64) as ex:
            for hit in ex.map(probe, [f"{net}.{h}" for h in range(1, 255)]):
                if hit:
                    return hit
    return None


base = f"http://{a.ip}:{a.port}"
try:
    st = requests.get(base + "/status", timeout=6).json()
except Exception:
    print(f"phone not reachable at {base}; looking for the Bitchat app on this network ...")
    found = find_phone(a.port)
    if not found:
        sys.exit("no Bitchat API found. Is 'Accept camera alerts' on and the app open?")
    base = f"http://{found}:{a.port}"
    st = requests.get(base + "/status", timeout=6).json()
    print(f"found the phone at {base}")
print(f"phone {base}  api_enabled={st.get('api_enabled')}  peers={st.get('peers_count')}")


def send_text(t):
    r = requests.post(base + "/send/text", json={"text": t + TAG, "category": "UNSET"}, timeout=15)
    if "Rate limited" in r.text:
        time.sleep(6)
        r = requests.post(base + "/send/text", json={"text": t + TAG, "category": "UNSET"}, timeout=15)
    return r


for d in json.load(open(APP / "data/disaster_images/descriptions.json", encoding="utf-8")):
    img = cv2.imread(str(APP / "data/disaster_images" / d["file"]))
    h, w = img.shape[:2]
    s = min(1.0, 640 / max(h, w))
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    r = requests.post(base + "/send/image", files={"image": (d["file"], jpg, "image/jpeg")}, timeout=20)
    print(d["file"], "image ->", r.status_code)
    time.sleep(6)

    desc = d["description"].strip()
    if a.short:
        desc = re.split(r"(?<=\.)\s", desc)[0]
    text = f"[{d['alert_level'].upper()}] {desc}"
    parts = textwrap.wrap(text, a.part)
    for k, p in enumerate(parts, 1):
        t = f"({k}/{len(parts)}) {p}" if len(parts) > 1 else p
        r = send_text(t)
        print(f"   text {k}/{len(parts)} {len(t)} chars -> {r.status_code}")
        time.sleep(6)
