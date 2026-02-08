"""
Shared test fixtures for Local Voice Reception AI.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure src is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_audio_int16():
    """Generate a sample int16 audio array (1 second, 16kHz mono)."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return sample_rate, audio


@pytest.fixture
def sample_audio_float32():
    """Generate a sample float32 audio array (3 seconds, 24kHz mono)."""
    sample_rate = 24000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return sample_rate, audio


@pytest.fixture
def mock_device_config():
    """Mock DeviceConfig for Apple Silicon."""
    import torch
    from src.utils.device import DeviceConfig, DeviceType

    return DeviceConfig(
        device="cpu",
        device_type=DeviceType.CPU,
        dtype=torch.float32,
        attn_implementation="sdpa",
    )


@pytest.fixture
def pronunciation_dict_path(tmp_path):
    """Create a temporary pronunciation dictionary."""
    dict_content = """
terms:
  - original: "Cor.Inc"
    reading: "コア インク"
    priority: high
  - original: "API"
    reading: "エーピーアイ"

patterns:
  - pattern: "([0-9]+)円"
    replacement: "\\\\1えん"
"""
    dict_path = tmp_path / "pronunciation_dict.yaml"
    dict_path.write_text(dict_content, encoding="utf-8")
    return str(dict_path)


@pytest.fixture
def ref_audio_file(tmp_path, sample_audio_float32):
    """Create a temporary reference audio WAV file (3 seconds)."""
    import soundfile as sf

    sr, audio = sample_audio_float32
    ref_path = tmp_path / "ref_voice.wav"
    sf.write(str(ref_path), audio, sr)
    return str(ref_path)


@pytest.fixture
def short_audio_file(tmp_path):
    """Create a very short reference audio WAV file (0.5 seconds)."""
    import soundfile as sf

    sr = 24000
    audio = np.sin(np.linspace(0, 1, int(sr * 0.5))).astype(np.float32)
    ref_path = tmp_path / "short_voice.wav"
    sf.write(str(ref_path), audio, sr)
    return str(ref_path)


@pytest.fixture
def mock_tts_model():
    """Create a mock Qwen3TTSModel."""
    model = MagicMock()
    model.generate_custom_voice.return_value = (
        [np.zeros(24000, dtype=np.float32)],
        24000,
    )
    model.generate_voice_clone.return_value = (
        [np.zeros(24000, dtype=np.float32)],
        24000,
    )
    model.create_voice_clone_prompt.return_value = {"cached": True, "language": "Japanese"}
    return model


@pytest.fixture
def base_config():
    """Base application configuration for testing."""
    return {
        "app": {"name": "Test", "version": "0.0.1"},
        "stt": {
            "model_path": "models/vosk/vosk-model-small-ja-0.22",
            "sample_rate": 16000,
        },
        "tts": {
            "mode": "custom_voice",
            "device": "cpu",
            "pronunciation_dict": "config/pronunciation_dict.yaml",
            "preload": False,
            "custom_voice": {
                "speaker": "ono_anna",
                "language": "Japanese",
            },
            "voice_clone": {
                "ref_audio": "data/voice_samples/company_voice.wav",
                "ref_text": "テスト音声です。",
            },
        },
        "llm": {
            "provider": "ollama",
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "qwen2.5:7b",
            },
            "temperature": 0.7,
            "max_tokens": 512,
        },
        "ui": {
            "host": "127.0.0.1",
            "port": 7860,
            "theme": "soft",
        },
    }
