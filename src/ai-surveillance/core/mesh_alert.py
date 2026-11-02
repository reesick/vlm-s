"""One alert = one image + its text, delivered to the Bitchat phone as a unit.

    from core.mesh_alert import MeshAlert, BitchatSender
    sender = BitchatSender.connect("172.20.10.4")          # finds the phone if the IP moved
    sender.send(MeshAlert(image=frame_bgr, text="...", alert_level="high"))

Delivery rules learned from the phone app (bitchat-android VlmMessageHandler /
AudiencePolicy):
  * The image goes as a file packet (POST /send/image).
  * The text goes right after it (POST /send/text). The app wraps API text in a
    camera "IDX1|S|" packet that only Gov/Command phones display, unless the
    text already contains "IDX1|". So every part carries the neutral tag
    " IDX1|N|{}|-": shown to every role, ignored by the gateway, and stripped
    from the chat bubble.
  * category "UNSET" keeps it in the "All" chip (not filed under Fire/Gov).
  * Text up to ~1500 chars goes as ONE message (800 chars verified on the phone).
  * The phone accepts one API message per rate window (5 s), so sends are
    spaced by rate_limit_ms + 1 s.
"""
from __future__ import annotations

import socket
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests

VISIBLE_TAG = " IDX1|N|{}|-"


@dataclass
class MeshAlert:
    """An image and the description that belongs to it."""
    image: np.ndarray                     # BGR frame
    text: str                             # description (any length; split on send)
    alert_level: str = ""                 # none|low|medium|high|critical (shown as a prefix)
    hazards: list[str] = field(default_factory=list)
    source: str = ""                      # e.g. file name or camera id (not sent)

    def headline(self) -> str:
        lvl = f"[{self.alert_level.upper()}] " if self.alert_level else ""
        return f"{lvl}{self.text.strip()}"

    def text_parts(self, max_chars: int = 1500) -> list[str]:
        # One text box per alert. Tested on the phone: an 800-char text with the
        # visible tag arrives as a single message (the earlier "long texts fail"
        # was the Gov-only filter, not size). Split only beyond max_chars.
        full = self.headline()
        if self.hazards:
            full += " | Hazards: " + ", ".join(self.hazards)
        chunks = textwrap.wrap(full, max_chars - 8) or [full[:max_chars]]
        if len(chunks) == 1:
            return chunks
        return [f"({i}/{len(chunks)}) {c}" for i, c in enumerate(chunks, 1)]

    def jpeg(self, max_dim: int = 480, quality: int = 60) -> bytes:
        # Small on purpose: a mesh image is split into ~469-byte fragments and one
        # lost fragment drops the whole image on that phone. 480 px / q60 is
        # ~15-25 KB (~40-55 fragments) instead of ~150 at 640 px / q70.
        img = self.image
        h, w = img.shape[:2]
        s = min(1.0, max_dim / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()


@dataclass
class SendResult:
    image_ok: bool
    text_ok: list[bool]

    @property
    def ok(self) -> bool:
        return self.image_ok and all(self.text_ok)


class BitchatSender:
    """Sends MeshAlerts to the Bitchat VLM API on a phone."""

    def __init__(self, base: str, gap_s: float = 6.0, timeout: float = 20.0,
                 image_settle_s: float = 14.0):
        self.base = base
        self.gap_s = gap_s
        self.timeout = timeout
        # A 40-70 KB image takes several seconds to fragment over BLE. Sending the
        # next message while it is still going out made some images or texts go
        # missing, so wait this long after each image before the next message.
        self.image_settle_s = image_settle_s
        self._last = 0.0
        self._next_ok = 0.0

    # ---- connection ---------------------------------------------------------
    @classmethod
    def connect(cls, ip: str = "", port: int = 8765, scan: bool = True) -> "BitchatSender":
        """Use ip if it answers, otherwise scan the laptop's /24 networks for the API."""
        cands = [ip] if ip else []
        for cand in cands:
            st = cls._status(f"http://{cand}:{port}")
            if st is not None:
                return cls._make(f"http://{cand}:{port}", st)
        if not scan:
            raise ConnectionError(f"Bitchat API not reachable at {ip}:{port}")
        print(f"[mesh] {ip or '(no ip)'}:{port} not reachable; scanning this network ...")
        found = cls.discover(port)
        if not found:
            raise ConnectionError("no Bitchat API found: is 'Accept camera alerts' on and the app open?")
        base = f"http://{found}:{port}"
        print(f"[mesh] found the phone at {base}")
        return cls._make(base, cls._status(base) or {})

    @classmethod
    def _make(cls, base: str, st: dict) -> "BitchatSender":
        gap = (st.get("rate_limit_ms", 5000) or 5000) / 1000.0 + 1.0
        print(f"[mesh] {base}  api_enabled={st.get('api_enabled')}  peers={st.get('peers_count')}  gap={gap:.1f}s")
        if st.get("api_enabled") is False:
            raise ConnectionError("phone reachable but 'Accept camera alerts' is off")
        return cls(base, gap_s=gap)

    @staticmethod
    def _status(base: str) -> dict | None:
        try:
            return requests.get(base + "/status", timeout=5).json()
        except Exception:
            return None

    @staticmethod
    def discover(port: int = 8765) -> str | None:
        nets = set()
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                addr = info[4][0]
                if not addr.startswith(("127.", "169.254.")):
                    nets.add(addr.rsplit(".", 1)[0])
        except OSError:
            pass

        def probe(addr):
            try:
                with socket.create_connection((addr, port), timeout=0.4):
                    pass
            except OSError:
                return None
            st = BitchatSender._status(f"http://{addr}:{port}")
            return addr if st and "api_enabled" in st else None

        for net in sorted(nets):
            with ThreadPoolExecutor(64) as ex:
                for hit in ex.map(probe, [f"{net}.{h}" for h in range(1, 255)]):
                    if hit:
                        return hit
        return None

    # ---- sending ------------------------------------------------------------
    def _wait(self):
        dt = max(self.gap_s - (time.perf_counter() - self._last), self._next_ok - time.perf_counter())
        if dt > 0:
            time.sleep(dt)

    def _post(self, path: str, **kw) -> requests.Response | None:
        for attempt in range(3):
            self._wait()
            try:
                r = requests.post(self.base + path, timeout=self.timeout, **kw)
            except requests.RequestException as e:
                print(f"[mesh]   {path} attempt {attempt + 1} failed: {e}")
                time.sleep(2)
                continue
            self._last = time.perf_counter()
            if "Rate limited" in r.text:
                continue
            return r
        return None

    def gateway_status(self) -> dict:
        """Command-centre gateway state on the phone (GET /gateway)."""
        try:
            return requests.get(self.base + "/gateway", timeout=6).json()
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    def configure_gateway(self, api_url: str, gateway_key: str = "", city_id: str = "") -> dict:
        """Turn on the phone's command-centre link (POST /gateway)."""
        body = {"enabled": True, "api_url": api_url}
        if gateway_key:
            body["gateway_key"] = gateway_key
        if city_id:
            body["city_id"] = city_id
        return requests.post(self.base + "/gateway", json=body, timeout=10).json()

    def send(self, alert: MeshAlert, command_centre: bool = True) -> SendResult:
        """Image first, then its text parts, each in its own rate window.

        Every message is a mesh broadcast with TTL 7: each phone that hears it
        relays it (PacketRelayManager), so it hops to every participant.
        command_centre: also send the description WITHOUT the visible tag; the
        app wraps that as an IDX1|S| camera packet, which the gateway phone
        forwards to the command centre (/api/v1/mesh/inbound) and Gov/Command
        phones display. Needs the gateway enabled on a phone with internet.
        """
        jpg = alert.jpeg()
        name = (alert.source or "alert").rsplit("\\", 1)[-1].rsplit("/", 1)[-1] or "alert.jpg"
        if not name.lower().endswith((".jpg", ".jpeg")):
            name += ".jpg"
        r = self._post("/send/image", files={"image": (name, jpg, "image/jpeg")})
        image_ok = bool(r is not None and r.status_code == 200)
        if image_ok:
            # settle time scales with size: ~4 s + 1 s per 5 KB (a 6 KB frame -> ~5 s)
            self._next_ok = time.perf_counter() + min(self.image_settle_s, 4.0 + len(jpg) / 5120)
        print(f"[mesh] image {name} {len(jpg) // 1024} KB -> {r.status_code if r is not None else 'failed'}")
        text_ok = []
        for part in alert.text_parts():
            r = self._post("/send/text", json={"text": part + VISIBLE_TAG, "category": "UNSET"})
            ok = bool(r is not None and r.status_code == 200)
            text_ok.append(ok)
            print(f"[mesh]   text {len(part)} chars -> {r.status_code if r is not None else 'failed'}")
        if command_centre:
            cc = alert.headline()[:1200]
            r = self._post("/send/text", json={"text": cc})
            ok = bool(r is not None and r.status_code == 200)
            text_ok.append(ok)
            print(f"[mesh]   command-centre copy (IDX1|S) -> {r.status_code if r is not None else 'failed'}")
        # let the text go out before the next alert's image starts
        self._next_ok = time.perf_counter() + self.gap_s
        return SendResult(image_ok, text_ok)
