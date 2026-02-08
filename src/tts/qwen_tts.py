"""
Qwen3-TTS Text-to-Speech Module.
Provides high-quality Japanese speech synthesis with voice cloning support.

Supports two modes:
- custom_voice: Preset speakers (e.g., Ono_Anna) via CustomVoice model
- voice_clone: Clone any voice from reference audio via Base model
"""

import logging
import re
import threading
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


# Model name mapping per mode
MODEL_MAP = {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


class QwenTTS:
    """
    Qwen3-TTS based Text-to-Speech engine.

    Supports:
    - Custom voice (preset speakers like Ono_Anna)
    - Voice cloning (from reference audio with prompt caching)
    - Voice design (natural language description)
    """

    # Available preset speakers for CustomVoice model
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
        model_name: Optional[str] = None,
        mode: str = "custom_voice",
        device: str = "auto",
        pronunciation_dict_path: Optional[str] = None,
    ):
        """
        Initialize Qwen TTS.

        Args:
            model_name: HuggingFace model name (auto-selected if None)
            mode: Operation mode ("custom_voice", "voice_clone", "voice_design")
            device: Device to use ("auto", "mps", "cuda", "cpu")
            pronunciation_dict_path: Path to pronunciation dictionary YAML
        """
        self.mode = mode
        self.model_name = model_name or MODEL_MAP.get(mode, MODEL_MAP["custom_voice"])
        self.model = None
        self.sample_rate = 24000  # Qwen3-TTS outputs 24kHz audio

        # Voice clone prompt cache (computed once, reused for every synthesis)
        self._voice_clone_prompt = None
        self._clone_language = "Japanese"
        self._prompt_lock = threading.RLock()
        self._model_lock = threading.Lock()

        # Device detection
        from src.utils.device import detect_device

        self.device_config = detect_device(device if device != "auto" else None)
        logger.info(f"TTS mode: {self.mode}, model: {self.model_name}")
        logger.info(f"TTS using device: {self.device_config}")

        # Load pronunciation dictionary
        self.pronunciation_dict = {}
        if pronunciation_dict_path:
            self._load_pronunciation_dict(pronunciation_dict_path)

    def _load_model(self):
        """Lazy load the TTS model (thread-safe against concurrent first-load)."""
        if self.model is not None:
            return

        with self._model_lock:
            # Double-check after acquiring lock
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

    def prepare_clone(
        self,
        ref_audio_path: str,
        ref_text: str,
        language: str = "Japanese",
    ):
        """
        Pre-compute voice clone prompt from reference audio.
        This caches the prompt so subsequent synthesize calls are faster
        and produce consistent voice output.

        Args:
            ref_audio_path: Path to reference audio file (3+ seconds recommended)
            ref_text: Transcript of the reference audio
            language: Language of the reference audio
        """
        self._load_model()

        ref_path = Path(ref_audio_path)
        if not ref_path.exists():
            logger.warning(f"Reference audio not found: {ref_path}")
            logger.warning("Voice cloning will fall back to custom_voice mode")
            return self

        # Validate reference audio duration
        import soundfile as sf_check

        data, sr_check = sf_check.read(str(ref_path))
        duration = len(data) / sr_check
        if duration < 1.0:
            logger.warning(f"Reference audio is very short ({duration:.1f}s). Minimum 1s required.")
            return self
        if duration < 3.0:
            logger.warning(
                f"Reference audio is {duration:.1f}s. 3+ seconds recommended for better quality."
            )

        if len(ref_text.strip()) < 5:
            logger.warning(f"Reference text is too short ({len(ref_text)} chars). 10+ recommended.")

        logger.info(f"Pre-computing voice clone prompt from: {ref_audio_path} ({duration:.1f}s)")
        logger.info(f"Reference text: {ref_text}")

        with self._prompt_lock:
            self._clone_language = language
            self._voice_clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(ref_path),
                ref_text=ref_text,
            )

        logger.info("Voice clone prompt cached successfully")
        return self

    def _load_pronunciation_dict(self, path: str):
        """Load pronunciation dictionary from YAML."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Pronunciation dict not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            logger.warning(f"Pronunciation dict is not a valid YAML mapping: {path}")
            return

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
        Synthesize speech from text using preset speaker (custom_voice mode).

        Args:
            text: Text to synthesize
            speaker: Preset speaker name (see SPEAKERS)
            language: Language ("Japanese", "Chinese", "English", etc.)
            instruct: Optional style instruction

        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        self._load_model()

        # Preprocess with pronunciation dictionary
        processed_text = self._preprocess_text(text)
        logger.debug(f"Preprocessed text: {processed_text}")

        # Always explicitly pass language to prevent auto-detection issues
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
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "Japanese",
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech using voice cloning.
        Uses cached voice_clone_prompt if available for consistent output.

        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio (ignored if prompt is cached)
            ref_text: Transcript of reference audio (ignored if prompt is cached)
            language: Language for synthesis output

        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        self._load_model()

        # Preprocess text
        processed_text = self._preprocess_text(text)
        logger.debug(f"Voice clone text: {processed_text}")

        # Use cached prompt if available (consistent voice across calls)
        with self._prompt_lock:
            cached_prompt = self._voice_clone_prompt
            cached_language = self._clone_language

        if cached_prompt is not None:
            # Always use the language that was set when the prompt was created
            # to ensure consistent Japanese output across all calls
            logger.debug(f"Using cached voice clone prompt (language={cached_language})")
            wavs, sr = self.model.generate_voice_clone(
                text=processed_text,
                language=cached_language,
                voice_clone_prompt=cached_prompt,
            )
            return wavs[0], sr

        # Fallback: compute from reference audio on-the-fly
        if ref_audio_path is None:
            raise ValueError(
                "No cached voice clone prompt and no ref_audio_path provided. "
                "Call prepare_clone() first or provide ref_audio_path."
            )

        ref_path = Path(ref_audio_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")

        logger.info(f"Computing voice clone from: {ref_audio_path} (not cached)")

        wavs, sr = self.model.generate_voice_clone(
            text=processed_text,
            language=language,
            ref_audio=str(ref_path),
            ref_text=ref_text or "",
        )

        return wavs[0], sr

    def update_reference_audio(
        self,
        ref_audio_path: str,
        ref_text: str,
        language: str = "Japanese",
    ):
        """
        Update the cached voice clone prompt with new reference audio.
        Use this when the user records a new voice sample.

        Args:
            ref_audio_path: Path to new reference audio file
            ref_text: Transcript of the new reference audio
            language: Language of the reference audio
        """
        with self._prompt_lock:
            self._voice_clone_prompt = None
        self.prepare_clone(ref_audio_path, ref_text, language)

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

    # Test custom_voice mode
    print("=== Testing custom_voice mode ===")
    tts_custom = QwenTTS(
        mode="custom_voice",
        pronunciation_dict_path="config/pronunciation_dict.yaml",
    )
    print("Available speakers:", tts_custom.list_speakers())
    print("\nSynthesizing test audio (custom_voice)...")

    audio, sr = tts_custom.synthesize(
        text="お電話ありがとうございます。コア株式会社でございます。",
        speaker="Ono_Anna",
        language="Japanese",
    )
    tts_custom.save_audio(audio, "test_custom_voice.wav", sr)
    print(f"Audio saved: test_custom_voice.wav (sample rate: {sr})")

    # Test voice_clone mode (requires reference audio)
    print("\n=== Testing voice_clone mode ===")
    ref_audio = "data/voice_samples/company_voice.wav"
    if Path(ref_audio).exists():
        tts_clone = QwenTTS(
            mode="voice_clone",
            pronunciation_dict_path="config/pronunciation_dict.yaml",
        )
        tts_clone.prepare_clone(
            ref_audio_path=ref_audio,
            ref_text="お電話ありがとうございます。コア株式会社でございます。",
            language="Japanese",
        )

        audio, sr = tts_clone.synthesize_with_clone(
            text="ご用件をお伺いいたします。",
            language="Japanese",
        )
        tts_clone.save_audio(audio, "test_voice_clone.wav", sr)
        print(f"Audio saved: test_voice_clone.wav (sample rate: {sr})")
    else:
        print(f"Reference audio not found: {ref_audio}")
        print("Record your voice and save it there to test voice cloning.")
