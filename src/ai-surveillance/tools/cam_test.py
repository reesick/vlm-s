"""Webcam test with logs.

    python tools/cam_test.py                # index 0, tries MSMF then DSHOW
    python tools/cam_test.py --index 1      # a different camera index
    python tools/cam_test.py --show         # also open a live preview (Q to quit)

Reports per backend: whether the device opened, how many frames were read in
3 s, fps, resolution, mean brightness (a black frame means shutter/privacy
cover or another app holding the camera) and saves a snapshot next to this
script as cam_<index>_<backend>.jpg.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

BACKENDS = {"MSMF": cv2.CAP_MSMF, "DSHOW": cv2.CAP_DSHOW, "ANY": cv2.CAP_ANY}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def test(index: int, name: str, show: bool) -> bool:
    log(f"--- index={index} backend={name} ---")
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(index, BACKENDS[name])
    log(f"open: isOpened={cap.isOpened()} in {time.perf_counter() - t0:.2f}s")
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    log(f"reported size={int(cap.get(3))}x{int(cap.get(4))} fps={cap.get(5)}")

    good = bad = 0
    last = None
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3.0:
        ok, frame = cap.read()
        if ok and frame is not None:
            good += 1
            last = frame
            if show:
                cv2.imshow(f"cam {index} {name}", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        else:
            bad += 1
            time.sleep(0.02)
    dt = time.perf_counter() - t0
    cap.release()
    cv2.destroyAllWindows()
    log(f"frames: good={good} failed={bad} in {dt:.1f}s -> {good / dt:.1f} fps")
    if last is None:
        log("RESULT: FAIL - opened but no frames (camera busy in another app, or wrong index)")
        return False
    bright = float(last.mean())
    out = Path(__file__).with_name(f"cam_{index}_{name}.jpg")
    cv2.imwrite(str(out), last)
    log(f"frame shape={last.shape} mean_brightness={bright:.1f} snapshot={out}")
    if bright < 8:
        log("RESULT: WARN - frames are black (shutter/privacy cover, or a virtual camera)")
        return False
    log("RESULT: OK")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--backend", choices=list(BACKENDS), default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    names = [a.backend] if a.backend else ["MSMF", "DSHOW"]
    results = {n: test(a.index, n, a.show) for n in names}
    log(f"SUMMARY index={a.index}: {results}")
    if any(results.values()):
        log("Use this in configs/pipeline.yaml ->  source: {type: webcam, path: %d}" % a.index)
