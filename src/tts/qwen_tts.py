"""
Qwen3-TTS Text-to-Speech Module.
Provides high-quality Japanese speech synthesis with voice cloning support.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

# Lazy import to avoid loading model at import time
Qwen3TTSModel = None


def _load_qwen_tts():
    """Lazy load Qwen TTS model class."""
    global Qwen3TTSModel
    if Qwen3TTSModel is None:
        from qwen_tts import Qwen3TTSModel as _Qwen3TTSModel

        Qwen3TTSModel = _Qwen3TTSModel
    return Qwen3TTSModel


class QwenTTS:
    """
    Qwen3-TTS based Text-to-Speech engine.

    Supports:
    - Custom voice (preset speakers like Ono_Anna)
    - Voice cloning (from reference audio)
    - Voice design (natural language description)
    """

    # Available preset speakers (lowercase with underscore)
    SPEAKERS = {
        "ono_anna": "Playful Japanese female voice, light nimble timbre",
        "vivian": "Bright, slightly edgy young female voice (Chinese native)",
        "serena": "Warm, gentle young female voice (Chinese native)",
        "ryan": "Dynamic male voice with strong rhythmic drive (English native)",
        "aiden": "Sunny American male voice with clear midrange",
        "dylan": "Male voice",
        "eric": "Male voice",
        "sohee": "Korean female voice",
        "uncle_fu": "Male voice",
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: str = "auto",
        pronunciation_dict_path: Optional[str] = None,
    ):
        """
        Initialize Qwen TTS.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use ("auto", "mps", "cuda", "cpu")
            pronunciation_dict_path: Path to pronunciation dictionary YAML
        """
        self.model_name = model_name
        self.model = None
        self.sample_rate = 24000  # Qwen3-TTS outputs 24kHz audio

        # Device detection
        from src.utils.device import detect_device

        self.device_config = detect_device(device if device != "auto" else None)
        logger.info(f"TTS using device: {self.device_config}")

        # Load pronunciation dictionary
        self.pronunciation_dict = {}
        if pronunciation_dict_path:
            self._load_pronunciation_dict(pronunciation_dict_path)

    def _load_model(self):
        """Lazy load the TTS model."""
        if self.model is not None:
            return

        logger.info(f"Loading Qwen TTS model: {self.model_name}")
        logger.info(f"Device config: {self.device_config}")

        _Qwen3TTSModel = _load_qwen_tts()

        self.model = _Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map=self.device_config.device,
            dtype=self.device_config.dtype,
            attn_implementation=self.device_config.attn_implementation,
        )

        logger.info("Qwen TTS model loaded successfully")

    def preload(self):
        """Preload the TTS model (call at startup to avoid first-request delay)."""
        self._load_model()
        return self

    def _load_pronunciation_dict(self, path: str):
        """Load pronunciation dictionary from YAML."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Pronunciation dict not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.pronunciation_dict = data
        logger.info(f"Loaded pronunciation dictionary: {len(data.get('terms', []))} terms")

    def _preprocess_text(self, text: str) -> str:
        """Apply pronunciation dictionary to text."""
        if not self.pronunciation_dict:
            return text

        # Apply term replacements
        for term in self.pronunciation_dict.get("terms", []):
            original = term.get("original", "")
            reading = term.get("reading", "")
            if original and reading:
                text = text.replace(original, reading)

        # Apply pattern replacements
        for pattern in self.pronunciation_dict.get("patterns", []):
            regex = pattern.get("pattern", "")
            replacement = pattern.get("replacement", "")
            if regex and replacement:
                text = re.sub(regex, replacement, text)

        return text

    def synthesize(
        self,
        text: str,
        speaker: str = "ono_anna",
        language: str = "Japanese",
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text using preset speaker.

        Args:
            text: Text to synthesize
            speaker: Preset speaker name (see SPEAKERS)
            language: Language ("Japanese", "Chinese", "English", etc.)
            instruct: Optional style instruction (e.g., "優しく話して")

        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        self._load_model()

        # Preprocess with pronunciation dictionary
        processed_text = self._preprocess_text(text)
        logger.debug(f"Preprocessed text: {processed_text}")

        # Generate audio
        wavs, sr = self.model.generate_custom_voice(
            text=processed_text,
            language=language,
            speaker=speaker,
            instruct=instruct or "",
        )

        return wavs[0], sr

    def synthesize_with_clone(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        language: str = "Japanese",
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech using voice cloning.

        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio file (3+ seconds)
            ref_text: Transcript of reference audio
            language: Language

        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        self._load_model()

        # Preprocess text
        processed_text = self._preprocess_text(text)

        # Load reference audio
        import soundfile as sf

        ref_audio, ref_sr = sf.read(ref_audio_path)

        # Generate with voice cloning
        wavs, sr = self.model.generate_voice_clone(
            text=processed_text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
        )

        return wavs[0], sr

    def save_audio(
        self,
        audio_data: np.ndarray,
        output_path: str,
        sample_rate: Optional[int] = None,
    ):
        """Save audio data to file."""
        import soundfile as sf

        sr = sample_rate or self.sample_rate
        sf.write(output_path, audio_data, sr)
        logger.info(f"Audio saved to: {output_path}")

    @classmethod
    def list_speakers(cls) -> dict:
        """List available preset speakers."""
        return cls.SPEAKERS.copy()


if __name__ == "__main__":
    # Test TTS
    logging.basicConfig(level=logging.INFO)

    tts = QwenTTS(
        pronunciation_dict_path="config/pronunciation_dict.yaml",
    )

    print("Available speakers:", tts.list_speakers())
    print("\nSynthesizing test audio...")

    audio, sr = tts.synthesize(
        text="お電話ありがとうございます。コア株式会社でございます。ご用件をお伺いいたします。",
        speaker="Ono_Anna",
        language="Japanese",
    )

    tts.save_audio(audio, "test_output.wav", sr)
    print(f"Audio saved: test_output.wav (sample rate: {sr})")
