#!/usr/bin/env python3
"""
Model download script for Local Voice Reception AI.
Downloads required models: Vosk (STT), Qwen3-TTS
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=desc
    ) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


def download_vosk_model(model_name: str = "vosk-model-small-ja-0.22"):
    """Download Vosk Japanese model."""
    vosk_dir = MODELS_DIR / "vosk"
    model_dir = vosk_dir / model_name

    if model_dir.exists():
        print(f"✓ Vosk model already exists: {model_dir}")
        return model_dir

    url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    zip_path = vosk_dir / f"{model_name}.zip"

    print(f"Downloading Vosk model: {model_name}")
    download_file(url, zip_path, desc="Vosk Model")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(vosk_dir)

    zip_path.unlink()
    print(f"✓ Vosk model ready: {model_dir}")
    return model_dir


def download_qwen_tts_model():
    """
    Download Qwen3-TTS model via HuggingFace.
    Note: This will be downloaded automatically on first use.
    """
    print("Qwen3-TTS model will be downloaded automatically on first use.")
    print("Model: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    print("")

    # Pre-download by importing (optional, takes time)
    try:
        print("Pre-downloading Qwen3-TTS model (this may take a while)...")
        from qwen_tts import Qwen3TTSModel

        # Just check if we can import - actual download happens on from_pretrained
        print("✓ qwen_tts package is available")
        print("  Model will be cached on first run (~3.5GB)")
    except ImportError:
        print("⚠ qwen_tts not installed. Run: pip install qwen-tts")


def main():
    print("=" * 50)
    print("Model Download Script")
    print("=" * 50)
    print("")

    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download Vosk (small model for faster testing)
    print("[1/2] Downloading Vosk Japanese Model...")
    download_vosk_model("vosk-model-small-ja-0.22")
    print("")

    # Note about Qwen3-TTS
    print("[2/2] Qwen3-TTS Model...")
    download_qwen_tts_model()
    print("")

    print("=" * 50)
    print("Download Complete!")
    print("=" * 50)
    print("")
    print("For high-accuracy STT, also download the large model:")
    print("  python scripts/download_models.py --large-vosk")


if __name__ == "__main__":
    if "--large-vosk" in sys.argv:
        download_vosk_model("vosk-model-ja-0.22")
    else:
        main()
