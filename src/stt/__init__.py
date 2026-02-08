"""Speech-to-Text module using Vosk."""

from .dictionary import STTDictionary
from .vosk_stt import VoskSTT

__all__ = ["VoskSTT", "STTDictionary"]
