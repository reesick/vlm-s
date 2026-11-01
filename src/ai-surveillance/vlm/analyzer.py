"""Fast, structured scene analysis on top of the shared Qwen2.5-VL model.

One `VLMAnalyzer.analyze()` call replaces the old two independent generate()
passes (scene prompt, then entity prompt, each re-encoding a ~768-token image).
Speed techniques, each switchable from configs/vlm.yaml `analyzer`:

  a. Visual token budget: the frame is resized here (INTER_AREA, sides are
     multiples of 28) so W*H <= image_max_pixels (default 200704 ~= 256 tokens).
  b. Set-of-Mark: tracker boxes + ids (P1, V1, ...) are drawn on the frame, so
     the model reports per-actor state in one pass instead of per-crop calls.
  c. KV-cache reuse: turn 2 (JSON report) continues from turn 1's returned
     past_key_values. Only the new text tokens are prefilled; the image is not
     re-encoded. Qwen2.5-VL keeps `model.rope_deltas` from the image turn and
     `_prepare_position_ids_for_generation` reuses it when the cache is
     non-empty, so positions stay correct for appended text.
  d. Scene-change cache: if the frame thumbnail, actor ids and context are
     unchanged, the previous result is returned without running the model.
  e. Short greedy outputs with robust JSON parsing (no keyword guessing).
  f. Warmup at init so the first real call does not pay CUDA kernel setup.

Frames are BGR (OpenCV) everywhere in the pipeline; conversion to RGB happens
right before the processor.
"""
from __future__ import annotations

import copy
import json
import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a hard dependency of the pipeline
    cv2 = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ALERT_LEVELS = ("none", "low", "medium", "high", "critical")
RISK_LEVELS = ("none", "low", "medium", "high")
KINDS = ("person", "vehicle", "boat", "animal", "object")
_KIND_PREFIX = {"person": "P", "vehicle": "V", "boat": "B", "animal": "A", "object": "O"}
_KIND_PRIORITY = {"person": 0, "vehicle": 1, "boat": 1, "animal": 1, "object": 2}
_KIND_COLOR_BGR = {
    "person": (0, 0, 255),
    "vehicle": (255, 128, 0),
    "boat": (255, 0, 255),
    "animal": (0, 200, 0),
    "object": (0, 255, 255),
}

_ALERT_SYNONYMS = {
    "no": "none", "safe": "none", "normal": "none", "minimal": "low", "minor": "low",
    "moderate": "medium", "med": "medium", "elevated": "medium", "severe": "high",
    "serious": "high", "danger": "high", "dangerous": "high", "extreme": "critical",
    "emergency": "critical",
}

DEFAULT_ANALYZER_CFG: dict[str, Any] = {
    "active_profile": "generic",
    "image_max_pixels": 200704,
    "escalated_max_pixels": 401408,
    "min_pixels": 3136,
    "scene_max_new_tokens": 96,
    "actors_max_new_tokens": 220,
    "kv_reuse": True,
    "mark_actors": True,
    "max_actors": 8,
    "skip_if_unchanged": True,
    "unchanged_threshold": 4.0,
    "max_cache_age_s": 30,
    "warmup": True,
    "json_prefill": True,
    "prompt_lookup_tokens": 10,
    "json_repetition_penalty": 1.0,
    "fast_decode": True,
    "static_cache_len": 2048,
    "scene_repetition_penalty": 1.05,
    "focus_blur": False,
    "focus_blur_ksize": 21,
    "focus_blur_pad": 0.15,
}

DEFAULT_PROFILES: dict[str, dict] = {
    "generic": {
        "description": "general scene understanding",
        "scene_prompt": (
            "Describe this camera frame in at most 2 short sentences (under 40 words): the "
            "place, the people, vehicles and objects, and what each is doing. Mention anything "
            "unusual or dangerous. If the scene is normal, say so."
        ),
        "hazard_types": ["fire", "smoke", "fall", "fight", "weapon", "crowd", "accident",
                         "flood", "intrusion", "abandoned_object", "other"],
        "actor_states": ["normal", "walking", "running", "standing", "sitting", "lying",
                         "fighting", "fallen", "in_vehicle", "unknown"],
    },
}


# ---------------------------------------------------------------------------
# Tracks -> actors
# ---------------------------------------------------------------------------

_VEHICLE_IDS = {1, 2, 3, 5, 7}
_ANIMAL_IDS = set(range(14, 24))


def coco_kind(cls_id: int) -> str:
    cls_id = int(cls_id)
    if cls_id == 0:
        return "person"
    if cls_id in _VEHICLE_IDS:
        return "vehicle"
    if cls_id == 8:
        return "boat"
    if cls_id in _ANIMAL_IDS:
        return "animal"
    return "object"


def _tget(track: Any, name: str, default: Any = None) -> Any:
    if isinstance(track, dict):
        return track.get(name, default)
    return getattr(track, name, default)


@dataclass
class Actor:
    id: str
    kind: str
    track_id: int
    xyxy: tuple[int, int, int, int]
    motion: str


class TrackMotionHistory:
    """Rolling per-track centre history used to describe motion in words.

    `update()` is cheap and meant to be called every frame from the main loop;
    `describe()` may be called from the VLM worker thread. A lock guards both.
    """

    STALE_S = 5.0

    def __init__(self, window_s: float = 2.0):
        self.window_s = float(window_s)
        self._hist: dict[int, deque] = {}
        self._last_seen: dict[int, float] = {}
        self._lock = threading.Lock()

    def update(self, tracks, t: float | None = None) -> None:
        if t is None:
            t = time.monotonic()
        t = float(t)
        with self._lock:
            for tr in tracks or ():
                tid = _tget(tr, "track_id")
                box = _tget(tr, "xyxy")
                if tid is None or box is None or int(tid) < 0:
                    continue
                tid = int(tid)
                x1, y1, x2, y2 = (float(v) for v in box[:4])
                dq = self._hist.get(tid)
                if dq is None:
                    dq = deque()
                    self._hist[tid] = dq
                dq.append((t, (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, y2 - y1)))
                self._last_seen[tid] = t
            cutoff = t - self.window_s
            for tid in list(self._hist):
                if t - self._last_seen.get(tid, t) > self.STALE_S:
                    self._hist.pop(tid, None)
                    self._last_seen.pop(tid, None)
                    continue
                dq = self._hist[tid]
                # keep at least the newest sample so describe() still has an anchor
                while len(dq) > 1 and dq[0][0] < cutoff:
                    dq.popleft()

    def describe(self, track_id: int) -> str:
        with self._lock:
            dq = self._hist.get(int(track_id))
            samples = list(dq) if dq else []
        if len(samples) < 2:
            return "unknown"
        t0, x0, y0, _ = samples[0]
        t1, x1, y1, _ = samples[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return "unknown"
        h = sum(s[3] for s in samples) / len(samples)
        dx, dy = x1 - x0, y1 - y0
        speed = math.hypot(dx, dy) / h / dt   # box heights per second
        if speed < 0.15:
            return "stationary"
        if abs(dx) >= abs(dy):
            direction = "moving right" if dx > 0 else "moving left"
        else:
            direction = "moving down (toward camera)" if dy > 0 else "moving up (away)"
        if speed < 0.6:
            bucket = "slow"
        elif speed < 1.5:
            bucket = "walking pace"
        else:
            bucket = "fast"
        return f"{direction}, {bucket}"


def actors_from_tracks(tracks, history: TrackMotionHistory | None, frame_shape,
                       max_actors: int = 8) -> list[Actor]:
    """Convert tracker output into numbered actors (P1.., V1.., ...).

    Persons first, then vehicles/boats/animals, then objects; within a group the
    largest boxes win when there are more than `max_actors`. Ids are assigned per
    kind in left-to-right order so they read naturally on the marked frame.
    """
    if not tracks:
        return []
    fh, fw = int(frame_shape[0]), int(frame_shape[1])
    cands = []
    for tr in tracks:
        tid = _tget(tr, "track_id")
        box = _tget(tr, "xyxy")
        if tid is None or box is None or int(tid) < 0:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
        x1, x2 = max(0, min(x1, fw - 1)), max(0, min(x2, fw - 1))
        y1, y2 = max(0, min(y1, fh - 1)), max(0, min(y2, fh - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cls_id = _tget(tr, "cls", None)
        kind = coco_kind(cls_id if cls_id is not None else -1)
        area = (x2 - x1) * (y2 - y1)
        cands.append((kind, int(tid), (x1, y1, x2, y2), area))
    cands.sort(key=lambda c: (_KIND_PRIORITY[c[0]], -c[3]))
    cands = cands[:max(0, int(max_actors))]
    cands.sort(key=lambda c: (c[2][0] + c[2][2]) / 2.0)   # left-to-right
    counters: dict[str, int] = {}
    actors: list[Actor] = []
    for kind, tid, box, _ in cands:
        counters[kind] = counters.get(kind, 0) + 1
        motion = history.describe(tid) if history is not None else "unknown"
        actors.append(Actor(id=f"{_KIND_PREFIX[kind]}{counters[kind]}", kind=kind,
                            track_id=tid, xyxy=box, motion=motion))
    actors.sort(key=lambda a: (_KIND_PRIORITY[a.kind], a.id[0], int(a.id[1:])))
    return actors


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class SceneAnalysis:
    scene: str
    alert_level: str
    hazards: list[dict]
    actors: list[dict]
    profile: str
    latency_s: float
    timings: dict
    cached: bool = False
    marked_frame: "np.ndarray | None" = None

    def to_entities(self) -> list[dict]:
        ents = [{
            "type": str(h.get("type", "other")),
            "confidence": float(h.get("confidence", 0.0)),
            "source": "vlm_detection",
            "description": str(h.get("description", "")),
        } for h in self.hazards]
        for a in self.actors:
            risk = a.get("risk")
            if risk in ("medium", "high"):
                ents.append({
                    "type": f"actor_{a.get('id')}",
                    "confidence": 0.85 if risk == "high" else 0.6,
                    "source": "vlm_actor",
                    "description": f"{a.get('id')} {a.get('kind')}: {a.get('action', '')}; {a.get('state', '')}",
                })
        return ents

    def to_message(self, max_len: int = 700) -> str:
        lines = [f"[{self.alert_level.upper()}] {self.scene.strip()}"]
        for a in self.actors:
            lines.append(f"{a.get('id')} {a.get('kind')} - {a.get('action', '')} "
                         f"({a.get('state', '')}) risk:{a.get('risk', '')}")
        if self.hazards:
            lines.append("Hazards: " + ", ".join(
                f"{h.get('type')} {float(h.get('confidence', 0.0)):.2f}" for h in self.hazards))
        msg = "\n".join(lines)
        if len(msg) > max_len:
            msg = msg[:max(0, max_len - 3)].rstrip() + "..."
        return msg

    def to_dict(self) -> dict:
        return {"scene": self.scene, "alert_level": self.alert_level,
                "hazards": self.hazards, "actors": self.actors, "profile": self.profile}


# ---------------------------------------------------------------------------
# Pure helpers (CPU, unit-checkable)
# ---------------------------------------------------------------------------

def budget_size(w: int, h: int, max_pixels: int, min_side: int = 56) -> tuple[int, int]:
    """Target (w, h): multiples of 28, w*h <= max_pixels, no upscaling except to min_side."""
    scale = min(1.0, math.sqrt(max_pixels / float(w * h))) if w * h > 0 else 1.0
    nw = max(min_side, int(w * scale) // 28 * 28)
    nh = max(min_side, int(h * scale) // 28 * 28)
    while nw * nh > max_pixels and (nw > min_side or nh > min_side):
        if nw >= nh and nw > min_side:
            nw -= 28
        elif nh > min_side:
            nh -= 28
        else:
            nw -= 28
    return nw, nh


def resize_for_budget(img: np.ndarray, max_pixels: int, min_side: int = 56) -> np.ndarray:
    h, w = img.shape[:2]
    nw, nh = budget_size(w, h, max_pixels, min_side)
    if (nw, nh) == (w, h):
        return img
    interp = cv2.INTER_AREA if nw * nh < w * h else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp)


def focus_blur(frame_bgr: np.ndarray, actors: list[Actor], ksize: int = 21,
               pad: float = 0.15) -> np.ndarray:
    """Blur everything outside the (padded) actor boxes; boxes stay sharp.

    Visual-prompting trick ("blur reverse mask", FGVP, NeurIPS 2023): the model
    attends to the sharp regions, so per-actor answers refer to the right
    person. Side effect: the JPEG sent over the mesh shrinks because the
    blurred background compresses well. The blur itself runs on a 1/4-size
    copy, so it costs ~1 ms. It does NOT change the visual token count; the
    token budget is set by `image_max_pixels`.
    """
    if not actors:
        return frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    k = max(3, int(ksize) | 1)
    small = cv2.resize(frame_bgr, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (k // 2 * 2 + 1, k // 2 * 2 + 1), 0)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros((h, w), dtype=np.uint8)
    for a in actors:
        x1, y1, x2, y2 = a.xyxy
        px, py = int((x2 - x1) * pad) + 4, int((y2 - y1) * pad) + 4
        cv2.rectangle(mask, (max(0, x1 - px), max(0, y1 - py)),
                      (min(w - 1, x2 + px), min(h - 1, y2 + py)), 255, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)[..., None].astype(np.float32) / 255.0
    return (frame_bgr * mask + bg * (1.0 - mask)).astype(np.uint8)


def draw_actor_marks(frame_bgr: np.ndarray, actors: list[Actor], target_scale: float) -> np.ndarray:
    """Draw boxes + id labels sized so they remain legible after downscaling by target_scale."""
    out = frame_bgr.copy()
    s = max(1e-3, float(target_scale))
    thick = max(2, int(math.ceil(2.5 / s)))
    font_scale = max(0.5, 0.55 / s)
    font_thick = max(1, int(math.ceil(1.6 / s)))
    for a in actors:
        x1, y1, x2, y2 = a.xyxy
        color = _KIND_COLOR_BGR.get(a.kind, (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        (tw, th), base = cv2.getTextSize(a.id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        ly = y1 - 4 if y1 - th - base - 4 >= 0 else min(out.shape[0] - 1, y1 + th + base + 4)
        cv2.rectangle(out, (x1, ly - th - base), (x1 + tw + 4, ly + base // 2), color, -1)
        cv2.putText(out, a.id, (x1 + 2, ly - base // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
    return out


def _repair_truncated_json(s: str) -> str | None:
    """Close a JSON object cut off by max_new_tokens at the last complete element."""
    stack: list[str] = []
    in_str = esc = False
    last_good = -1
    last_good_stack: list[str] = []
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
            if not stack:
                return s[:i + 1]
            last_good, last_good_stack = i, list(stack)
    if last_good < 0:
        return None
    closing = "".join("}" if c == "{" else "]" for c in reversed(last_good_stack))
    return s[:last_good + 1] + closing


def parse_json_object(text: str) -> dict | None:
    """Extract the outermost JSON object from model text; None if impossible."""
    raw = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)(?:```|$)", raw, re.DOTALL)
    if fence and "{" in fence.group(1):
        raw = fence.group(1).strip()
    start = raw.find("{")
    if start < 0:
        return None
    end = raw.rfind("}")
    candidates = []
    if end > start:
        candidates.append(raw[start:end + 1])
    repaired = _repair_truncated_json(raw[start:])
    if repaired:
        candidates.append(repaired)
    for c in candidates:
        for variant in (c, re.sub(r",\s*([}\]])", r"\1", c)):
            try:
                obj = json.loads(variant)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _norm_level(value: Any, allowed: tuple[str, ...], default: str) -> str:
    v = str(value or "").strip().lower()
    if v in allowed:
        return v
    v = _ALERT_SYNONYMS.get(v, v)
    if v in allowed:
        return v
    if v == "critical" and "critical" not in allowed:
        return "high"
    return default


def _clamp01(x: Any) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f):
        return 0.0
    return max(0.0, min(1.0, f))


def normalize_report(obj: dict, actors: list[Actor] | None) -> tuple[str, list[dict], list[dict]]:
    alert = _norm_level(obj.get("alert_level"), ALERT_LEVELS, "unknown")
    hazards = []
    for h in obj.get("hazards") or []:
        if not isinstance(h, dict) or not h.get("type"):
            continue
        hazards.append({
            "type": str(h.get("type")).strip().lower().replace(" ", "_"),
            "confidence": _clamp01(h.get("confidence", 0.0)),
            "description": str(h.get("description", "")).strip(),
        })
    known = {a.id: a for a in (actors or [])}
    out_actors = []
    seen = set()
    for a in obj.get("actors") or []:
        if not isinstance(a, dict):
            continue
        # The prompt allows actors doing the same thing to share one entry
        # ("id": "P1,P2,P4") to save output tokens; expand it here.
        ids = [s for s in re.split(r"[,\s/&]+", str(a.get("id", "")).strip().upper()) if s]
        for aid in ids:
            if aid in seen:
                continue
            seen.add(aid)
            kind = known[aid].kind if aid in known else str(a.get("kind", "object")).strip().lower()
            if kind not in KINDS:
                kind = "object"
            out_actors.append({
                "id": aid,
                "kind": kind,
                "state": str(a.get("state", "unknown")).strip().lower().replace(" ", "_") or "unknown",
                "action": str(a.get("action", "")).strip(),
                "risk": _norm_level(a.get("risk"), RISK_LEVELS, "low"),
            })
    return alert, hazards, out_actors


def thumb_gray(frame_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_bgr, (64, 36), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small.astype(np.float32)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class VLMAnalyzer:
    """Two-turn (scene text -> JSON report) analysis sharing one KV cache."""

    def __init__(self, model, processor, cfg: dict, profiles: dict,
                 profile: str | None = None, device: str = "cuda:0"):
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.cfg = {**DEFAULT_ANALYZER_CFG, **(cfg or {})}
        self.profiles = dict(DEFAULT_PROFILES)
        self.profiles.update(profiles or {})
        self.device = device
        self._lock = threading.Lock()
        self._kv_fail_streak = 0
        self._kv_warned = False
        self._kv_disabled = False
        self._last: SceneAnalysis | None = None
        self._last_key: tuple | None = None
        self._last_thumb: np.ndarray | None = None
        self._last_t = 0.0
        self.profile_name = ""
        self.set_profile(profile or self.cfg.get("active_profile") or "generic")
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        # All VLM GPU work runs on its own stream so the detector's kernels on the
        # default stream are not queued behind seconds of VLM work.
        self._stream = (torch.cuda.Stream(device=self._model_device())
                        if torch is not None and torch.cuda.is_available() else None)
        self._gd = None
        if (self.cfg.get("fast_decode") and model is not None and torch is not None
                and torch.cuda.is_available()
                and not any(type(m).__name__ == "Linear8bitLt" for m in model.modules())):
            try:
                from vlm.fast_decode import GraphDecoder
                eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None) or []
                eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
                self._gd = GraphDecoder(model, eos + [self._im_end_id],
                                        max_cache_len=int(self.cfg.get("static_cache_len", 2048)),
                                        device=self._model_device())
            except Exception as e:  # noqa: BLE001
                print(f"[VLMAnalyzer] WARN: fast decoder unavailable ({type(e).__name__}: {e}); "
                      "using HF generate")
                self._gd = None
        if self.cfg.get("warmup") and model is not None:
            self.warmup()

    # -- profile -----------------------------------------------------------
    def set_profile(self, name: str) -> None:
        if name not in self.profiles:
            raise ValueError(f"[VLMAnalyzer] unknown profile {name!r}; "
                             f"available: {sorted(self.profiles)}")
        if name != self.profile_name:
            self.profile_name = name
            self._last = None   # cached result was produced under another profile

    @property
    def profile(self) -> dict:
        p = {**DEFAULT_PROFILES["generic"], **self.profiles[self.profile_name]}
        return p

    # -- low-level generate helpers ---------------------------------------
    def _model_device(self):
        dev = getattr(self.model, "device", None)
        return dev if dev is not None else self.device

    def _encode(self, prompt: str, rgb: np.ndarray | None):
        content = []
        if rgb is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        kwargs = {"text": [text], "return_tensors": "pt"}
        if rgb is not None:
            kwargs["images"] = [rgb]
        return self.processor(**kwargs).to(self._model_device())

    def _generate(self, max_new_tokens: int, **inputs):
        kwargs = dict(max_new_tokens=int(max_new_tokens), do_sample=False, use_cache=True,
                      return_dict_in_generate=True, pad_token_id=self.tokenizer.pad_token_id)
        kwargs.update(inputs)
        return self.model.generate(**kwargs)

    def _decode(self, ids) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def warmup(self) -> None:
        if self.model is None:
            return
        t0 = time.perf_counter()
        gray = np.full((112, 112, 3), 128, dtype=np.uint8)
        with self._on_stream(), torch.inference_mode():
            inputs = self._encode("Hi", gray)
            self._generate(2, **inputs)
            if self._gd is not None:
                try:
                    self._gd.warmup(inputs)
                    print(f"[VLMAnalyzer] CUDA-graph decoder captured in {self._gd.capture_s:.2f}s "
                          f"(static KV cache {self._gd.max_len} tokens)")
                except Exception as e:  # noqa: BLE001
                    print(f"[VLMAnalyzer] WARN: CUDA-graph capture failed ({type(e).__name__}: {e}); "
                          "using HF generate")
                    self._gd = None
        print(f"[VLMAnalyzer] warmup done in {time.perf_counter() - t0:.2f}s")

    def _on_stream(self):
        import contextlib
        return torch.cuda.stream(self._stream) if self._stream is not None else contextlib.nullcontext()

    def _wait_stream(self) -> None:
        """Wait for queued VLM work with the GIL released (polling, not a blocking sync)."""
        if self._stream is None:
            return
        evt = torch.cuda.Event()
        evt.record(self._stream)
        while not evt.query():
            time.sleep(0.001)

    @property
    def fast_decode_active(self) -> bool:
        return self._gd is not None and self._gd.graph is not None

    def _disable_fast(self, e: Exception) -> None:
        print(f"[VLMAnalyzer] WARN: fast decode failed ({type(e).__name__}: {e}); "
              "falling back to HF generate for this session")
        self._gd = None

    # -- prompts -------------------------------------------------------------
    def _json_prefill(self) -> str:
        """Start of the assistant answer, forced so the model emits compact JSON
        (no ``` fence, no indentation: ~40% fewer output tokens)."""
        return '{"alert_level": "' if self.cfg.get("json_prefill", True) else ""

    def _scene_prompt(self, actors: list[Actor], marked: bool) -> str:
        p = self.profile["scene_prompt"].strip()
        if marked and actors:
            ids = ", ".join(a.id for a in actors)
            p += (f" Coloured boxes labelled {ids} mark tracked people/vehicles; "
                  "refer to them by label.")
        return p

    def _report_prompt(self, actors: list[Actor], extra_context: str) -> str:
        prof = self.profile
        max_actors = int(self.cfg.get("max_actors", 8))
        lines = ["Now give a structured report of this frame."]
        if actors:
            lines.append("Tracked actors (labelled boxes in the image):")
            lines += [f"{a.id} {a.kind}, tracker motion: {a.motion}" for a in actors]
            lines.append("Report every listed actor. Actors in the same state doing the same "
                         'thing share ONE entry with ids joined by commas, e.g. "id": "P1,P2".')
        else:
            lines.append(f"No tracker boxes are available. List up to {max_actors} visible "
                         "people or vehicles yourself with ids A1, A2, ...; actors doing the same "
                         'thing share ONE entry, e.g. "id": "A1,A2".')
        if extra_context:
            lines.append(f"Sensor context: {extra_context.strip()}")
        lines.append("Hazard types: " + ", ".join(prof["hazard_types"]))
        lines.append("Actor states: " + ", ".join(prof["actor_states"]))
        lines.append("Alert levels: " + ", ".join(ALERT_LEVELS))
        lines.append("Only report hazards you can actually see; use an empty list if none.")
        lines.append("Keep it short: action at most 3 words, description at most 4 words.")
        lines.append("Answer with compact single-line JSON only (no code fence), exactly this schema:")
        if actors:
            # kind is known from the tracker; not asking for it saves ~6 tokens per actor
            actor_schema = ('{"id": "<id>", "state": "<actor state>", '
                            '"action": "<what it is doing, few words>", "risk": "none|low|medium|high"}')
        else:
            actor_schema = ('{"id": "<id>", "kind": "person|vehicle|boat|animal|object", '
                            '"state": "<actor state>", "action": "<what it is doing, few words>", '
                            '"risk": "none|low|medium|high"}')
        lines.append('{"alert_level": "<alert level>", "hazards": [{"type": "<hazard type>", '
                     '"confidence": <0.0-1.0>, "description": "<short phrase>"}], '
                     '"actors": [' + actor_schema + ']}')
        return "\n".join(lines)

    # -- cache ---------------------------------------------------------------
    def _cache_hit(self, thumb, key) -> bool:
        if not self.cfg.get("skip_if_unchanged") or self._last is None:
            return False
        if self._last_key != key or self._last_thumb is None:
            return False
        if time.monotonic() - self._last_t > float(self.cfg.get("max_cache_age_s", 30)):
            return False
        diff = float(np.mean(np.abs(thumb - self._last_thumb)))
        return diff < float(self.cfg.get("unchanged_threshold", 4.0))

    # -- main entry ----------------------------------------------------------
    def analyze(self, frame_bgr, actors: list[Actor] | None = None, extra_context: str = "",
                force: bool = False, max_pixels: int | None = None) -> SceneAnalysis:
        t_start = time.perf_counter()
        actors = list(actors or [])
        extra_context = extra_context or ""
        thumb = thumb_gray(frame_bgr)
        key = (self.profile_name, tuple(a.id for a in actors), extra_context)

        with self._lock:
            if not force and self._cache_hit(thumb, key):
                prev = self._last
                timings = dict(prev.timings)
                timings["cached"] = True
                return replace(prev, hazards=copy.deepcopy(prev.hazards),
                               actors=copy.deepcopy(prev.actors), timings=timings,
                               cached=True, latency_s=time.perf_counter() - t_start)

            with self._on_stream():
                result = self._analyze_uncached(frame_bgr, actors, extra_context, max_pixels)
                self._wait_stream()
            result.latency_s = time.perf_counter() - t_start
            self._last, self._last_key, self._last_thumb = result, key, thumb
            self._last_t = time.monotonic()
            return result

    def _analyze_uncached(self, frame_bgr, actors, extra_context, max_pixels) -> SceneAnalysis:
        budget = int(max_pixels or self.cfg.get("image_max_pixels", 200704))
        h, w = frame_bgr.shape[:2]
        nw, nh = budget_size(w, h, budget)
        marked = bool(self.cfg.get("mark_actors")) and bool(actors)
        base = frame_bgr
        blurred = bool(self.cfg.get("focus_blur")) and bool(actors)
        if blurred:
            base = focus_blur(frame_bgr, actors, int(self.cfg.get("focus_blur_ksize", 21)),
                              float(self.cfg.get("focus_blur_pad", 0.15)))
        src = draw_actor_marks(base, actors, nw / float(w)) if marked else base.copy()
        small = resize_for_budget(src, budget)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image_tokens = (small.shape[0] // 28) * (small.shape[1] // 28)

        use_reuse = bool(self.cfg.get("kv_reuse")) and not self._kv_disabled
        timings: dict[str, Any] = {"turn1_s": 0.0, "turn2_s": 0.0, "image_tokens": image_tokens,
                                   "kv_reused": False, "cached": False, "focus_blur": blurred,
                                   "resized_to": (small.shape[1], small.shape[0])}

        scene, raw = "", None
        if self._gd is not None:
            try:
                scene, raw = self._fast_two_turn(rgb, actors, marked, extra_context, use_reuse, timings)
            except Exception as e:  # noqa: BLE001 - fall back to the HF path
                self._disable_fast(e)
                raw = None
        if raw is None:
            scene, raw = self._hf_two_turn(rgb, actors, marked, extra_context, use_reuse, timings)

        obj = parse_json_object(raw)
        if obj is None:
            print(f"[VLMAnalyzer] WARN: could not parse JSON report; raw text was: {raw!r}")
            alert, hazards, out_actors = "unknown", [], []
        else:
            alert, hazards, out_actors = normalize_report(obj, actors)
        timings["raw_report"] = raw
        return SceneAnalysis(scene=scene, alert_level=alert, hazards=hazards, actors=out_actors,
                             profile=self.profile_name, latency_s=0.0, timings=timings,
                             cached=False, marked_frame=src if marked else None)

    def _turn2_suffix_ids(self, actors, extra_context, device):
        suffix = ("\n<|im_start|>user\n" + self._report_prompt(actors, extra_context)
                  + "<|im_end|>\n<|im_start|>assistant\n" + self._json_prefill())
        return self.tokenizer(suffix, add_special_tokens=False,
                              return_tensors="pt").input_ids.to(device)

    def _fast_two_turn(self, rgb, actors, marked, extra_context, use_reuse, timings):
        """CUDA-graph decode over one static KV cache (vlm/fast_decode.py).

        Turn 1 prefills image + scene prompt and decodes the description.
        Turn 2 appends the report prompt to the SAME cache (kv_reuse) so the
        image is neither re-encoded nor re-prefilled, then decodes the JSON.
        """
        gd = self._gd
        if gd.graph is None:              # e.g. warmup disabled: capture on a throwaway prompt
            with torch.inference_mode():
                gd.warmup(self._encode("Hi", None))
        n_new2 = max(int(self.cfg.get("actors_max_new_tokens", 220)), 70 + 20 * len(actors))
        rp1 = float(self.cfg.get("scene_repetition_penalty", 1.05) or 1.0)
        rp2 = float(self.cfg.get("json_repetition_penalty", 1.0) or 1.0)
        with torch.inference_mode():
            t0 = time.perf_counter()
            inputs = self._encode(self._scene_prompt(actors, marked), rgb)
            n_prompt = int(inputs["input_ids"].shape[1])
            t_enc = time.perf_counter()
            first = gd.prefill(inputs, penalty=rp1)
            t_pre = time.perf_counter()
            toks1 = gd.decode(first, int(self.cfg.get("scene_max_new_tokens", 96)))
            scene = self._decode(toks1)
            timings["turn1_s"] = round(time.perf_counter() - t0, 3)
            timings["preprocess_s"] = round(t_enc - t0, 3)
            timings["prefill1_s"] = round(t_pre - t_enc, 3)
            timings["prompt_tokens"] = n_prompt
            timings["turn1_new_tokens"] = len(toks1)

            t1 = time.perf_counter()
            dev = inputs["input_ids"].device
            suffix_ids = self._turn2_suffix_ids(actors, extra_context, dev)
            last = toks1[-1]
            head = ([last] if last not in gd.eos else []) + [self._im_end_id]
            head_ids = torch.tensor([head], dtype=torch.long, device=dev)
            if use_reuse:
                first2 = gd.extend(torch.cat([head_ids, suffix_ids], dim=1), penalty=rp2)
                timings["kv_reused"] = True
                ctx = int(gd.cur)
            else:
                body = torch.tensor([toks1[:-1]], dtype=torch.long, device=dev)
                full = torch.cat([inputs["input_ids"], body.view(1, -1), head_ids, suffix_ids], dim=1)
                first2 = gd.prefill(self._fresh_inputs(inputs, full), penalty=rp2)
                ctx = int(full.shape[1])
            toks2 = gd.decode(first2, n_new2)
            raw = self._json_prefill() + self._decode(toks2)
            timings["turn2_s"] = round(time.perf_counter() - t1, 3)
            timings["turn2_context_tokens"] = ctx
            timings["turn2_new_tokens"] = len(toks2)
            dec_s = timings["turn1_s"] - timings["prefill1_s"] - timings["preprocess_s"] + timings["turn2_s"]
            timings["decode_tok_per_s"] = round((len(toks1) + len(toks2)) / max(1e-6, dec_s), 1)
            timings["decoder"] = "cuda_graph"
        return scene, raw

    def _hf_two_turn(self, rgb, actors, marked, extra_context, use_reuse, timings):
        """Reference path: HF generate() with a DynamicCache returned from turn 1."""
        with torch.inference_mode():
            # ---- turn 1: image + scene prompt -> short description ----------
            t0 = time.perf_counter()
            inputs = self._encode(self._scene_prompt(actors, marked), rgb)
            n_prompt = inputs["input_ids"].shape[1]
            out1 = self._generate(self.cfg.get("scene_max_new_tokens", 96), **inputs)
            scene = self._decode(out1.sequences[0, n_prompt:])
            timings["turn1_s"] = round(time.perf_counter() - t0, 3)
            timings["prompt_tokens"] = int(n_prompt)
            timings["turn1_new_tokens"] = int(out1.sequences.shape[1] - n_prompt)

            # ---- turn 2: JSON report continuing the same conversation ------
            seq = out1.sequences
            if int(seq[0, -1]) != self._im_end_id:
                seq = torch.cat([seq, seq.new_tensor([[self._im_end_id]])], dim=1)
            suffix = ("\n<|im_start|>user\n" + self._report_prompt(actors, extra_context)
                      + "<|im_end|>\n<|im_start|>assistant\n" + self._json_prefill())
            suffix_ids = self.tokenizer(suffix, add_special_tokens=False,
                                        return_tensors="pt").input_ids.to(seq.device)
            full = torch.cat([seq, suffix_ids], dim=1)
            past = getattr(out1, "past_key_values", None)
            del out1

            t1 = time.perf_counter()
            out2 = None
            lookup = int(self.cfg.get("prompt_lookup_tokens", 0) or 0)
            extra2: dict[str, Any] = {"prompt_lookup_num_tokens": lookup} if lookup > 0 else {}
            rp = self.cfg.get("json_repetition_penalty")
            if rp is not None:
                extra2["repetition_penalty"] = float(rp)
            # Cap grows with the actor count so long actor lists are not cut off.
            n_new2 = max(int(self.cfg.get("actors_max_new_tokens", 220)),
                         70 + 20 * len(actors))
            if use_reuse and past is not None:
                try:
                    out2 = self._generate(
                        n_new2,
                        input_ids=full, attention_mask=torch.ones_like(full),
                        past_key_values=past, **extra2)
                    timings["kv_reused"] = True
                    self._kv_fail_streak = 0
                except Exception as e:  # noqa: BLE001 - any failure falls back to a fresh pass
                    self._kv_fail_streak += 1
                    if not self._kv_warned:
                        print(f"[VLMAnalyzer] WARN: KV-cache reuse failed ({type(e).__name__}: {e}); "
                              "falling back to a fresh full pass")
                        self._kv_warned = True
                    if self._kv_fail_streak >= 2:
                        self._kv_disabled = True
                        print("[VLMAnalyzer] WARN: KV-cache reuse disabled for this session "
                              "after 2 consecutive failures")
                    out2 = None
            del past
            if out2 is None:
                out2 = self._generate(n_new2, **self._fresh_inputs(inputs, full), **extra2)
            raw = self._json_prefill() + self._decode(out2.sequences[0, full.shape[1]:])
            timings["turn2_s"] = round(time.perf_counter() - t1, 3)
            timings["turn2_context_tokens"] = int(full.shape[1])
            timings["turn2_new_tokens"] = int(out2.sequences.shape[1] - full.shape[1])
            del out2
        timings["decoder"] = "hf_generate"
        return scene, raw

    @staticmethod
    def _fresh_inputs(inputs, full) -> dict:
        """Same token sequence as the reuse path, but re-encoding the image (no cache)."""
        fresh = {}
        L = inputs["input_ids"].shape[1]
        for k, v in inputs.items():
            if k == "input_ids":
                fresh[k] = full
            elif k == "attention_mask":
                fresh[k] = torch.ones_like(full)
            elif torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == L and k != "image_grid_thw":
                pad = v.new_zeros((v.shape[0], full.shape[1] - L))
                fresh[k] = torch.cat([v, pad], dim=1)   # e.g. mm_token_type_ids: text = 0
            else:
                fresh[k] = v
        return fresh

    # -- whole-video verdict ---------------------------------------------------
    def summarize_timeline(self, entries: list[tuple[float, SceneAnalysis]],
                           max_new_tokens: int = 160) -> str:
        if not entries:
            return ""
        entries = sorted(entries, key=lambda e: e[0])
        if len(entries) > 20:
            step = len(entries) / 20.0
            entries = [entries[int(i * step)] for i in range(20)]
        lines = []
        for t, a in entries:
            acts = "; ".join(f"{x.get('id')} {x.get('kind')}: {x.get('action', '')} "
                             f"({x.get('state', '')}, risk {x.get('risk', '')})" for x in a.actors[:6])
            haz = ", ".join(h.get("type", "") for h in a.hazards)
            lines.append(f"t={t:.1f}s [{a.alert_level}] {a.scene.strip()[:240]}"
                         + (f" | actors: {acts}" if acts else "")
                         + (f" | hazards: {haz}" if haz else ""))
        prompt = (
            f"These are timestamped observations from one camera ({self.profile['description']}).\n"
            + "\n".join(lines)
            + "\n\nIn 2-4 sentences, summarize what happened overall, who was at risk and how "
              "the situation changed. Start with 'OVERALL: <LEVEL>.' where LEVEL is one of "
            + ", ".join(ALERT_LEVELS) + "."
        )
        with self._lock, self._on_stream(), torch.inference_mode():
            inputs = self._encode(prompt, None)
            n = inputs["input_ids"].shape[1]
            if self._gd is not None:
                try:
                    if self._gd.graph is None:
                        self._gd.warmup(self._encode("Hi", None))
                    first = self._gd.prefill(inputs, penalty=float(
                        self.cfg.get("scene_repetition_penalty", 1.05) or 1.0))
                    return self._decode(self._gd.decode(first, max_new_tokens))
                except Exception as e:  # noqa: BLE001
                    self._disable_fast(e)
            out = self._generate(max_new_tokens, **inputs)
            return self._decode(out.sequences[0, n:])
