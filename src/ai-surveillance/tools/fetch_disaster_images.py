"""Download a small set of disaster photos from Wikimedia Commons (with licence info).

  python tools/fetch_disaster_images.py [--per-topic 1] [--out data/disaster_images]

Writes the images plus sources.json (title, page URL, author, licence) so every
photo sent to the phone can be attributed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

APP_DIR = Path(__file__).resolve().parents[1]
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "kurukshetra-hackathon-disaster-demo/1.0 (offline rescue VLM demo)"}
TOPICS = [
    ("flood", "flooded street people water rescue"),
    ("earthquake", "earthquake collapsed building rubble rescue"),
    ("fire", "car fire firefighter"),
    ("landslide", "landslide road debris"),
    ("cyclone", "cyclone damage houses destroyed"),
    ("wildfire", "wildfire smoke forest fire"),
    ("flood_rescue", "flood rescue boat"),
    ("building_fire", "building fire firefighters"),
]
EXTRA = [
    ("flooded_street", "flooded street cars submerged"),
    ("landslide2", "landslide house destroyed"),
    ("hurricane", "hurricane damage aerial"),
    ("wildfire2", "wildfire flames hillside"),
    ("collapsed", "collapsed building earthquake"),
]
OK_LICENCES = re.compile(r"^(cc0|public domain|pd|cc by(-sa)? [0-9.]+)", re.I)


def _get(url, **kw):
    """GET with retries: this network intermittently resets TLS handshakes."""
    last = None
    for attempt in range(6):
        try:
            r = requests.get(url, headers=UA, timeout=60, **kw)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def search(query: str, n: int) -> list[dict]:
    r = _get(API, params={
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": 25,
        "prop": "imageinfo", "iiprop": "url|mime|size|extmetadata", "iiurlwidth": 800})
    pages = sorted((r.json().get("query", {}).get("pages", {}) or {}).values(), key=lambda p: p.get("index", 0))
    out = []
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata", {})
        lic = re.sub("<[^>]+>", "", meta.get("LicenseShortName", {}).get("value", "")).strip()
        if ii.get("mime") not in ("image/jpeg", "image/png") or not OK_LICENCES.match(lic):
            continue
        if ii.get("width", 0) < 640:
            continue
        out.append({"title": p["title"], "url": ii.get("thumburl") or ii["url"],
                    "page": ii.get("descriptionurl"), "licence": lic,
                    "author": re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()[:120]})
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=1)
    ap.add_argument("--out", default="data/disaster_images")
    ap.add_argument("--topics", default="", help="comma list of topic tags to fetch (default: all)")
    a = ap.parse_args()
    want = {t.strip() for t in a.topics.split(",") if t.strip()}
    out = APP_DIR / a.out
    out.mkdir(parents=True, exist_ok=True)
    src_file = out / "sources.json"
    sources = json.loads(src_file.read_text()) if src_file.exists() else []
    have = {x["title"] for x in sources}
    for tag, q in TOPICS + EXTRA:
        if want and tag not in want:
            continue
        try:
            hits = search(q, a.per_topic)
        except Exception as e:  # noqa: BLE001
            print(f"[fetch] {tag}: search failed: {e}")
            continue
        for k, h in enumerate(hits):
            if h["title"] in have:
                continue
            ext = ".png" if h["url"].lower().endswith(".png") else ".jpg"
            fn = out / f"{len(sources) + 1:02d}_{tag}{ext}"
            try:
                r = _get(h["url"])
                fn.write_bytes(r.content)
            except Exception as e:  # noqa: BLE001
                print(f"[fetch] {tag}: download failed: {e}")
                continue
            sources.append({"file": fn.name, "topic": tag, **h})
            print(f"[fetch] {fn.name}  {h['licence']}  {h['title']}")
            time.sleep(4.0)        # upload.wikimedia.org rate-limits bursts (HTTP 429)
    (out / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(f"[fetch] {len(sources)} images in {out}")
    return 0 if sources else 1


if __name__ == "__main__":
    sys.exit(main())
