"""Send the same photo in several formats, each preceded by a label text.

  python tools/image_send_test.py [--ip 172.20.10.4]

Look at the receiving phone: for every "TEST n ..." label, note whether the
picture after it appears.
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import cv2

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from core.mesh_alert import BitchatSender, VISIBLE_TAG  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ip", default="172.20.10.4")
a = ap.parse_args()
s = BitchatSender.connect(a.ip)
img = cv2.imread(sorted(glob.glob(str(APP / "data/disaster_images/*.jpg")))[1])


def enc(dim, q, ext=".jpg", opt=False):
    h, w = img.shape[:2]
    k = min(1.0, dim / max(h, w))
    im = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    if ext == ".png":
        return cv2.imencode(".png", im, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tobytes()
    p = [cv2.IMWRITE_JPEG_QUALITY, q] + ([cv2.IMWRITE_JPEG_OPTIMIZE, 1] if opt else [])
    return cv2.imencode(".jpg", im, p)[1].tobytes()


tests = [
    ("1 old format 640px q70 (the one that showed before)", "/send/image", enc(640, 70), "image/jpeg", {}),
    ("2 small 360px q50", "/send/image", enc(360, 50), "image/jpeg", {}),
    ("3 tiny 240px q40", "/send/image", enc(240, 40), "image/jpeg", {}),
    ("4 small 360px q50 via /send/analysis", "/send/analysis", enc(360, 50), "image/jpeg", {}),
    ("5 small 320px PNG", "/send/image", enc(320, 0, ".png"), "image/png", {}),
]
for label, path, data, mime, extra in tests:
    r = s._post("/send/text", json={"text": f"TEST {label}: image comes next" + VISIBLE_TAG, "category": "UNSET"})
    print(f"TEST {label}\n   label -> {r.status_code if r is not None else 'failed'}")
    time.sleep(2)
    name = "test" + label.split()[0] + (".png" if mime == "image/png" else ".jpg")
    r = s._post(path, files={"image": (name, data, mime)}, data=extra)
    print(f"   image {len(data)} bytes via {path} -> {r.status_code if r is not None else 'failed'} {r.text[:60] if r is not None else ''}")
    time.sleep(8 + len(data) / 4096)       # let the whole image go out before the next label
print("done: check which TEST labels have a picture right after them")
