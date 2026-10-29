"""Download the Qwen2.5-VL-3B-Instruct VLM weights to the local Hugging Face cache.

Called by start_surveillance.ps1 step 2. Skips the download entirely when the
model is already present in the HF cache, so re-running the launcher is cheap.
"""

import sys

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"  # matches configs/vlm.yaml


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[vlm] huggingface_hub not installed - pip install from requirements passes it.")
        return 1

    print(f"[vlm] checking / downloading {MODEL_ID} (~7.1 GB on first run) ...")
    path = snapshot_download(MODEL_ID)
    print(f"[vlm] ready at: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
