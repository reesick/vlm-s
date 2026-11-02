# KURUKSHETRA / INDRADHANU

**Offline disaster surveillance: a camera feed is described by a local vision-language model, and the picture plus its description is shared over a Bitchat Bluetooth mesh, phone to phone, without internet.**

```text
 Camera (laptop webcam / phone camera / video file)
        |
        v
 Qwen2.5-VL-3B (NF4, CUDA graph decoder, optional QLoRA adapter)  ->  description on screen
        |
        v  HTTP (same Wi-Fi)
 Bitchat phone ("API phone")  --BLE mesh, TTL 7-->  every Bitchat phone nearby
        |
        v  (optional, when a phone has internet)
 Command centre  (/api/v1/mesh/inbound)
```

---

## Quick start (one file)

**Needs:** Windows 10/11, NVIDIA GPU with 6 GB+ VRAM, [Python 3.11](https://www.python.org/downloads/), about 15 GB free disk, internet for the first run.

```powershell
git clone https://github.com/reesick/vlm-s.git
cd vlm-s
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

`start.ps1` creates the virtual environment, installs the packages, downloads the Qwen2.5-VL-3B weights (about 7 GB, once) and starts the live camera. The first run takes 20 to 45 minutes. Later runs start in under a minute.

| Command | What it does |
| --- | --- |
| `.\start.ps1` | Live camera, VLM description on screen, image + description sent to the Bitchat mesh |
| `.\start.ps1 -Camera 0` | Use the laptop webcam (default `2` = phone camera such as a Samsung A36) |
| `.\start.ps1 -PhoneIp 192.168.1.5` | Phone IP. Leave it out and the script finds the phone on your Wi-Fi |
| `.\start.ps1 -NoShare` | Describe on screen only, send nothing |
| `.\start.ps1 -Mode images` | Send sample disaster photos and their VLM descriptions to the phone |
| `.\start.ps1 -Mode pipeline` | Full stack: YOLO detection, tracking, pose, events, VLM, alerts |
| `.\start.ps1 -Mode setup` | Only install and download, run nothing |
| `.\start.ps1 -CcUrl https://... -CcKey ...` | Also set the phone's command-centre gateway |

Keys in the live window: **S** or **Space** sends the current frame now, **Q** quits.

---

## Phone setup (Bitchat)

1. Install `apk/bitchat-indradhanu-debug.apk` on the phone that will receive from the laptop (the "API phone"). Other mesh phones can run the same APK.
   If an older Bitchat build signed with a different key is installed, uninstall it first.
2. Open the app, set a nickname and role, and keep Bluetooth on.
3. Settings → **Accept camera alerts** → ON. The app now serves its API on port `8765`.
4. Connect the laptop and the phone to the **same Wi-Fi or hotspot**.
5. Run `.\start.ps1`. It finds the phone by itself. You can also pass `-PhoneIp`.

Optional command centre: phone Settings → **Command Centre** → address, gateway key, city, "Sync when online".

**How delivery works**
- The laptop compresses each frame to about 5–8 KB (360 px JPEG), because BLE carries it in ~469-byte fragments.
- The text is sent with a visible tag (` IDX1|N|{}|-`), so every role sees it. Untagged camera packets are Gov/Command-only in this app.
- On connect, the laptop sets the phone's API rate limit to 0 and spaces the messages itself, because the app drops images that arrive inside its rate window without saying so.
- Every message is a mesh broadcast with TTL 7, so each phone relays it to the next.

---

## Repository layout

```text
.
├── start.ps1                 # ONE file: setup + run (start here)
├── run.ps1                   # full pipeline launcher with logs (used by -Mode pipeline)
├── go.ps1                    # full pipeline on a phone camera, waits for it to come up
├── apk/                      # ready-to-install Bitchat APK (debug build of src/bitchat-android)
├── scripts/                  # send_to_phone.ps1, fix_pagefile.ps1
├── docs/                     # results/ (benchmarks), notes/ (older fix notes)
├── screenshots/
└── src/
    ├── ai-surveillance/      # Python app
    │   ├── run_vlm_to_bitchat.py   # live camera -> VLM -> screen + mesh (start.ps1 default)
    │   ├── send_vlm_alerts.py      # folder of images -> VLM -> mesh
    │   ├── annotate_video.py       # video -> annotated MP4 with boxes + descriptions
    │   ├── run_flood_video.py      # offline flood-video report
    │   ├── run_vlm_only.py         # VLM on the camera, no sending
    │   ├── pipeline/               # full pipeline (python -m pipeline.main_loop)
    │   ├── core/                   # detection, tracking, pose, events, mesh_alert.py (image+text)
    │   ├── vlm/                    # Qwen loader, analyzer, CUDA-graph decoder, KV cache
    │   ├── configs/                # pipeline.yaml, models.yaml, vlm.yaml, roi.yaml
    │   ├── tools/                  # qlora_train.py, fetch_disaster_images.py, export_onnx.py,
    │   │                           # image_send_test.py, bitchat_debug/ (connection helpers)
    │   ├── benchmarks/             # speed, VRAM, quantization evaluation
    │   ├── tests/                  # pytest suite
    │   └── models/                 # weights (NOT in git, see below)
    └── bitchat-android/      # Android app source (Kotlin)
```

---

## Models (not in git)

| Model | How you get it |
| --- | --- |
| Qwen2.5-VL-3B-Instruct | Downloaded by `start.ps1` into the Hugging Face cache |
| YOLOv8n, YOLOv8n-pose, ONNX exports | Downloaded / exported by `run.ps1` (`-Mode pipeline`) |
| Face models (insightface `buffalo_s`) | Downloaded by insightface on first use |
| Fire/smoke and smoking YOLO (`models/*.pt`) | Optional fine-tuned weights. Without them the pipeline uses heuristics |
| QLoRA flood adapter `models/lora/flood_v1` | Optional. Train it with `tools/qlora_train.py`. Without it the base model is used |

---

## VLM details

- **Quantization:** NF4 with double quantization (about 2.25 GB VRAM). INT8 was about 6× slower with no accuracy gain.
- **Speed:** static KV cache with a CUDA-graph decoder runs at about 37–45 tokens/s. One description takes about 4–6 s on an RTX 4050 Laptop GPU (it was 15.6 s before).
- **Set-of-Mark:** detector boxes are drawn and numbered on the frame, so the description can refer to "person 2" and so on.
- **QLoRA:** the adapter is trained on the frozen NF4 model and kept unmerged (merging it into NF4 rounds the change away). Results are in `docs/results/`.

Settings live in `src/ai-surveillance/configs/vlm.yaml`.

---

## Manual use (without start.ps1)

```powershell
cd src\ai-surveillance
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe run_vlm_to_bitchat.py --camera 0            # live
.\.venv\Scripts\python.exe send_vlm_alerts.py                           # image folder
.\.venv\Scripts\python.exe annotate_video.py <video.mp4> --show         # annotated video
.\.venv\Scripts\python.exe -m pipeline.main_loop --video <file> --profile flood
.\.venv\Scripts\python.exe -m pytest tests -q                          # tests
```

Build the APK yourself (JDK 21 and the Android SDK):

```powershell
cd src\bitchat-android
.\gradlew.bat :app:assembleDebug -x lint
```

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `running scripts is disabled` | Run `powershell -ExecutionPolicy Bypass -File .\start.ps1` |
| `no Bitchat API found` | The phone and laptop must be on the same Wi-Fi, the app open, and **Accept camera alerts** ON |
| Phone found but `Accept camera alerts is off` | Turn it on in the app's Settings |
| Camera window black / camera not found | Try `-Camera 0` or `-Camera 1`. For a phone camera, open its camera/webcam app first |
| Text arrives, picture shows `[Image]` | The phone runs an older Bitchat build. Install `apk/bitchat-indradhanu-debug.apk` |
| Pictures arrive late on other phones | Normal. BLE sends them in fragments, and each relay hop adds a few seconds |
| CUDA out of memory | Close other GPU apps. The VLM needs about 2.5 GB, the full pipeline about 4 GB |
| Windows error 1455 (paging file) | Run `scripts\fix_pagefile.ps1` as administrator, then reboot |

---

## Responsible use

This is a hackathon prototype. A person should check every alert before acting on it. Secure the Wi-Fi and the phone API, and follow local law on cameras and consent.

## Licences

- Project code: MIT unless a file says otherwise.
- Ultralytics (YOLO): AGPL-3.0. Qwen2.5-VL: its own licence on Hugging Face.
- Bitchat Android: the upstream project licence (see `src/bitchat-android`).
- Sample disaster images are fetched from Wikimedia Commons. Their licences are written to `data/disaster_images/sources.json`.
