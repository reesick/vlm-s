# KURUKSHETRA

## AI-powered surveillance with real-time mesh alerts

KURUKSHETRA is a modular security-camera platform that turns a live camera feed into actionable, explainable events. It combines fast local computer vision with pose analysis, identity tracking, a vision-language model, structured event storage, and Bitchat mesh notifications to deliver alerts directly to an Android device.

> **Built for the Kurukshetra Hackathon** | Python surveillance pipeline + Android Bitchat companion

<p align="center">
	<strong>Detect</strong>&nbsp;&nbsp;•&nbsp;&nbsp;
	<strong>Understand</strong>&nbsp;&nbsp;•&nbsp;&nbsp;
	<strong>Verify</strong>&nbsp;&nbsp;•&nbsp;&nbsp;
	<strong>Alert</strong>
</p>

---

## Why KURUKSHETRA?

Traditional camera systems record everything and explain nothing. KURUKSHETRA keeps the camera local, watches for meaningful changes, and turns those changes into structured events:

- **Fast local detection** for people, phones, fire, smoke, falls, fights, crowds, smoking, violence, and abandoned objects.
- **Persistent identity context** using tracking, re-identification, and face recognition.
- **Natural-language scene understanding** through Qwen2.5-VL-3B-Instruct.
- **Evidence attached to alerts** with keyframes and optional event clips.
- **Offline-first alert delivery** through the Bitchat mesh network instead of a cloud dashboard.
- **Structured history** in SQLite for later inspection, querying, and auditing.

## System at a glance

```text
Camera / video / RTSP
					|
					v
	Frame Router + motion gate
					|
					+--> YOLO detection --> ByteTrack --> identity + attributes
					|
					+--> Pose --> fall / fight analysis
					|
					+--> Event detectors --> SQLite + keyframes + clips
					|
					+--> VLM escalation --> scene description + entities
					|
					v
	Bitchat HTTP API --> Android mesh alert
```

The pipeline is deliberately layered. Lightweight stages run continuously, expensive stages are scheduled or triggered only when useful, and priority events such as fire, falls, and fights can immediately escalate to the VLM and notification channel.

## Project status

| Area | Status | Role |
| --- | --- | --- |
| YOLO detection, tracking, and live display | Complete | Baseline real-time perception |
| Pose and fall detection | Complete | Keypoint-based posture analysis |
| ONNX acceleration | Complete | Lower inference overhead |
| Re-identification and face recognition | Complete | Persistent identity context |
| Fire, smoke, smoking, phone, crowd, violence, and object-left events | Complete | Situation-specific detection |
| PAR, ROI zones, event buffering, and SQLite logging | Complete | Context and auditability |
| Fight analysis and event clip writing | Complete | Higher-level temporal reasoning |
| Qwen2.5-VL-3B scene understanding | Live | Natural-language verification |
| Bitchat mesh alerts | Live | Android notifications and keyframes |

## Repository layout

```text
.
├── src/
│   ├── ai-surveillance/       # Python perception and alerting pipeline
│   │   ├── configs/            # Model, pipeline, ROI, and VLM configuration
│   │   ├── core/               # Detection, tracking, pose, identity, events
│   │   ├── pipeline/           # Frame routing and main orchestration loop
│   │   ├── vlm/                # VLM inference, escalation, caching, merging
│   │   ├── tests/               # Unit, smoke, and integration tests
│   │   ├── tools/               # ONNX export and face enrollment utilities
│   │   └── models/              # Runtime model weights, intentionally ignored
│   └── bitchat-android/        # Android companion project and integration assets
├── data/                       # Runtime SQLite data, keyframes, and exports
├── docs/                       # Project documentation
├── screenshots/                # Demonstration media
├── LICENSE/                    # License and legal material
└── requirements.txt            # Root dependency placeholder
```

## Quick start

### 1. Enter the surveillance application

```bash
cd src/ai-surveillance
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the pinned application dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python -m pipeline.main_loop
```

Other useful modes:

```bash
# Camera + VLM scene descriptions, without the full detector stack
python run_vlm_only.py

# VLM scene descriptions delivered through Bitchat
python run_vlm_to_bitchat.py

# Discover and diagnose the Android Bitchat API
python diagnose_connection.py --ip <ANDROID_IP>
```

Press `Q` or `Esc` in the video window to stop the live pipeline. In VLM-only mode, press `Space` to force an inference pass.

## Configure the pipeline

The main configuration lives in `src/ai-surveillance/configs/`:

- `pipeline.yaml` controls the video source, feature flags, scheduling, storage, and Bitchat delivery.
- `models.yaml` defines model paths, thresholds, and detector settings.
- `vlm.yaml` controls the Qwen model, device, quantization, and inference behavior.
- `roi.yaml` defines regions of interest.

Example feature configuration:

```yaml
features:
	pose: true
	fall_detection: true
	fight: true
	reid: true
	face: true
	fire_smoke: true
	smoking: true
	phone: true
	gathering: true
	violence: false
	object_left: true
	par: true
	vlm: true
```

## Android and Bitchat alerts

KURUKSHETRA uses the Bitchat Android companion as a local alert surface. The surveillance machine and phone should be connected to the same Wi-Fi network.

1. Install and open the Bitchat Android application.
2. Enable its VLM/API integration in **Settings → VLM API Settings**.
3. Find the phone's Wi-Fi IP address.
4. Configure `configs/pipeline.yaml`:

```yaml
bitchat:
	enabled: true
	ip: "<ANDROID_IP>"
	port: 8765
	channel: "surveillance"
	rate_limit_s: 5.5
	send_keyframes: true
	send_scene: true
```

5. Verify connectivity:

```bash
python diagnose_connection.py --ip <ANDROID_IP>
```

High-priority events bypass normal rate limiting. Alerts can include a message, a camera keyframe, and, for selected events, a recorded clip.

| Event | Example alert |
| --- | --- |
| Fire or smoke | `[FIRE] Fire detected in camera view` |
| Fall | `[FALL] Person 1 has fallen` |
| Fight | `[FIGHT] Fight between persons detected` |
| Phone use | `[PHONE] Person 1 using a phone` |
| Gathering | `[GATHERING] 4 people gathered` |
| VLM scene | `[CAM] Natural-language scene description` |

## Detection and reasoning stack

| Component | Responsibility |
| --- | --- |
| YOLO | Person, object, fire, smoke, smoking, and phone detection |
| ByteTrack | Stable short-term track IDs |
| Pose model | Keypoints for posture, fall, and fight analysis |
| OneEuro / EMA smoothing | Stabilizes noisy keypoints over time |
| OSNet + FAISS | Visual re-identification across frames |
| SCRFD + MobileFaceNet | Face detection and embeddings |
| PAR | Pedestrian attribute recognition |
| Motion gate | Avoids expensive work on unchanged frames |
| Qwen2.5-VL-3B | Scene descriptions and structured entity extraction |
| SQLite | Thread-safe event and keyframe persistence |
| Bitchat | Local mesh delivery to Android |

## Performance target

The reference target is an RTX 4050 Laptop GPU with 6 GB VRAM:

| Metric | Typical value |
| --- | --- |
| Detector-only camera rate | ~18–25 FPS |
| Camera rate while VLM runs | ~18–25 FPS |
| VLM inference | ~10–15 seconds per pass |
| VLM memory with NF4 quantization | ~2.3–2.5 GB VRAM |

The VLM runs in a background thread so its slower inference does not block the camera loop.

## Testing

From `src/ai-surveillance`:

```bash
python -m pytest tests/ -v
```

The test suite includes detector logic, fall and fight behavior, event buffering, Bitchat integration, VLM checks, and phase smoke tests. Hardware-dependent tests may require a camera, model weights, GPU, or a reachable Android device.

## Model and runtime data policy

Model weights, virtual environments, generated databases, keyframes, videos, and local logs are intentionally excluded from source control. See `src/ai-surveillance/models/README.md` for the expected model layout and place downloaded weights under that directory before enabling the corresponding feature.

## Responsible use

This project is a hackathon security prototype. Validate detections before acting on them, secure the local network and API endpoint, obtain appropriate consent for camera and face processing, and follow the laws and policies that apply to your deployment. Automated alerts should support human review, not replace it.

## License notes

- Project code: MIT, unless a component states otherwise.
- Ultralytics components and associated models: AGPL-3.0.
- The D-Fire model and other third-party assets retain their original licenses.

Review third-party licenses before distributing a deployment that bundles model weights.

## Acknowledgements

KURUKSHETRA brings together open-source work across real-time vision, tracking, pose estimation, model quantization, and local mesh communication. The project is organized so those components can be tuned independently as the prototype evolves.