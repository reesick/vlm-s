"""Direct link to the Indradhanu / DisruptionOps command centre over the internet.

Every VLM description of this camera is POSTed, with its frame and the camera's
position, to ``<api>/api/v1/ingest/camera`` (header ``X-Mesh-Gateway-Key``):

* the command centre logs every frame (``camera.frame``) and keeps the picture, so
  operators see what the edge camera sees even when nothing is wrong;
* a description that names a hazard (fire, smoke, flood, people trapped or
  stranded, collapse, injury, fallen tree) becomes a report there, with the frame
  attached, and goes through the same trust / dedup / planning as any report.

The Bitchat mesh path (``core.mesh_alert``) is unchanged and stays the offline
fallback: when this laptop has no internet the phone carries the alert.

Position: a camera frame has no GPS, so the camera's position is given on the
command line (``--lat/--lon``). ``--demo-pin`` uses a fixed point inside the PCCOE
drone map instead and tells the command centre it is a demo position, so the
incident is drawn where the drone swarm can fly and is labelled as such.

Clips: the camera loop hands every frame to ``push_frame``; the last few seconds
are kept small (320 px, ~4 fps). When a description names a hazard, that buffer is
encoded to a short MP4 (H.264 ``avc1``, falling back to VP8 WebM, then ``mp4v``; browsers play the first two)
and sent as ``clipBase64`` with the frame, so the audit trail can play what the
camera saw in the seconds before it raised the alarm. ``--no-clip`` turns it off.

Sends happen on a background thread, newest first; a description is never held
up waiting for the network, and an old frame is dropped rather than queued for
minutes behind a slow link.
"""
from __future__ import annotations

import base64
import collections
import os
import queue
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass

import cv2
import requests

#: Inside the drone swarm's PCCOE map (ward w-pc-01), about 60 m north-west of its anchor.
DEMO_PIN = (18.65300, 73.76190)


@dataclass
class Frame:
    image: object
    text: str
    alert_level: str
    hazards: list
    at: float
    clip: tuple[str, str] | None = None


class CommandCentreLink:
    def __init__(self, api: str, gateway_key: str, *, lat: float, lon: float, node_id: str = "vlm-cam-01",
                 label: str | None = None, demo_position: bool = False, city_id: str = "pune",
                 max_px: int = 640, quality: int = 70, timeout: float = 12.0,
                 clip_seconds: float = 6.0, clip_fps: float = 4.0, clip_px: int = 320):
        self.api = api.rstrip("/")
        if self.api.endswith("/api/v1"):
            self.api = self.api[: -len("/api/v1")]
        self.key = gateway_key
        self.lat, self.lon = float(lat), float(lon)
        self.node = node_id[:60]
        self.label = label or f"VLM camera {node_id}"
        self.demo = bool(demo_position)
        self.city = city_id
        self.max_px, self.quality, self.timeout = max_px, quality, timeout
        self.sent = self.reports = self.failed = 0
        self.last: dict = {}
        self.clip_seconds, self.clip_fps, self.clip_px = clip_seconds, clip_fps, clip_px
        self._ring: collections.deque = collections.deque(maxlen=max(1, int(clip_seconds * clip_fps)))
        self._ring_t = 0.0
        self._ring_lock = threading.Lock()
        self._q: queue.Queue = queue.Queue(maxsize=2)
        threading.Thread(target=self._worker, daemon=True, name="command-centre").start()
        where = f"{self.lat:.5f}, {self.lon:.5f}" + (" (DEMO POSITION)" if self.demo else "")
        print(f"[cc] direct link on: {self.api}/api/v1/ingest/camera  node={self.node}  at {where}")

    # -------------------------------------------------------------- input --
    def push_frame(self, image) -> None:
        """Every camera frame comes through here; a few per second are kept for the clip."""
        if self.clip_seconds <= 0:
            return
        now = time.time()
        if now - self._ring_t < 1.0 / self.clip_fps:
            return
        self._ring_t = now
        try:
            h, w = image.shape[:2]
            s = min(1.0, self.clip_px / float(max(h, w)))
            small = cv2.resize(image, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
        except Exception:  # noqa: BLE001
            return
        with self._ring_lock:
            self._ring.append(small)

    def _clip(self) -> tuple[str, str] | None:
        """The buffered seconds as a data URL, or None (no frames / no codec)."""
        with self._ring_lock:
            frames = list(self._ring)
        if len(frames) < 4:
            return None
        h, w = frames[0].shape[:2]
        for fourcc, ext, mime in (("avc1", ".mp4", "video/mp4"), ("VP80", ".webm", "video/webm"), ("mp4v", ".mp4", "video/mp4")):
            fd, path = tempfile.mkstemp(suffix=ext)
            os.close(fd)
            try:
                vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), self.clip_fps, (w, h))
                if not vw.isOpened():
                    continue
                for fr in frames:
                    if fr.shape[:2] == (h, w):
                        vw.write(fr)
                vw.release()
                data = open(path, "rb").read()
                if 1000 < len(data) <= 3_400_000:
                    return f"data:{mime};base64," + base64.b64encode(data).decode(), mime
            except Exception:  # noqa: BLE001
                pass
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        return None

    def offer(self, image, text: str, alert_level: str = "", hazards: list | None = None) -> None:
        """Queue one described frame (non-blocking; the oldest waiting frame is dropped)."""
        if not text:
            return
        clip = self._clip() if (hazards or (alert_level or "").lower() in ("warning", "critical", "high", "alert")) else None
        item = Frame(image=image, text=text, alert_level=alert_level or "", hazards=list(hazards or []), at=time.time(), clip=clip)
        while True:
            try:
                self._q.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass

    def check(self) -> bool:
        """Is the command centre reachable and the key accepted? (heartbeat, no report filed)"""
        try:
            r = requests.post(f"{self.api}/api/v1/mesh/gateway/heartbeat", headers={"X-Mesh-Gateway-Key": self.key},
                              json={"gatewayId": self.node, "cityId": self.city, "label": self.label,
                                    "lat": self.lat, "lon": self.lon, "appVersion": "vlm-s"}, timeout=40)
            print(f"[cc] command centre check: HTTP {r.status_code}"
                  + ("" if r.status_code == 200 else f" {r.text[:160]}"))
            return r.status_code == 200
        except requests.RequestException as e:
            print(f"[cc] command centre unreachable ({type(e).__name__}); the mesh stays the only path")
            return False

    # ------------------------------------------------------------- sender --
    def _jpeg_b64(self, image) -> str | None:
        try:
            h, w = image.shape[:2]
            s = min(1.0, self.max_px / float(max(h, w)))
            img = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else image
            q = self.quality
            while q >= 35:
                ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
                if ok and len(buf) <= 350_000:
                    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
                q -= 15
        except Exception as e:  # noqa: BLE001
            print(f"[cc] could not encode the frame: {e}")
        return None

    def _post(self, f: Frame) -> bool:
        body = {"nodeId": self.node, "eventId": f"{self.node}-{int(f.at)}-{uuid.uuid4().hex[:6]}"[:60],
                "lat": self.lat, "lon": self.lon, "caption": f.text[:1200], "alertLevel": f.alert_level or None,
                "hazards": [str(h) for h in f.hazards][:12], "confidence": 0.75, "demoPosition": self.demo,
                "label": self.label, "occurredAt": int(f.at), "cityId": self.city,
                "imageBase64": self._jpeg_b64(f.image)}
        if f.clip:
            body["clipBase64"], body["clipMime"] = f.clip
        for attempt in (1, 2):
            try:
                r = requests.post(f"{self.api}/api/v1/ingest/camera", headers={"X-Mesh-Gateway-Key": self.key},
                                  json=body, timeout=self.timeout if attempt == 1 else 40)
                if r.status_code == 200:
                    out = r.json() or {}
                    self.sent += 1
                    self.reports += out.get("outcome") in ("report", "linked")
                    self.last = out
                    what = {"report": "NEW REPORT", "linked": "added to an incident", "held": "held for an officer",
                            "logged": "logged"}.get(out.get("outcome"), out.get("outcome"))
                    print(f"[cc] frame -> command centre: {what}"
                          + (f" ({out.get('kind')})" if out.get("kind") else "")
                          + (f" ward {out.get('wardId')}" if out.get("wardId") else "")
                          + (f" - {out.get('reason')}" if out.get("reason") else ""))
                    return True
                print(f"[cc] command centre answered HTTP {r.status_code}: {r.text[:160]}")
                if r.status_code in (400, 401, 403, 404, 422):
                    break                                 # retrying will not change the answer
            except requests.RequestException as e:
                print(f"[cc] send failed ({type(e).__name__}){'; retrying once (server may be waking up)' if attempt == 1 else ''}")
        self.failed += 1
        return False

    def _worker(self) -> None:
        while True:
            f = self._q.get()
            if time.time() - f.at > 120:                   # stale: a newer view is more useful
                continue
            self._post(f)
