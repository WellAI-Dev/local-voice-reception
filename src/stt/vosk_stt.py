"""
Vosk Speech-to-Text Module.
Provides offline speech recognition using Vosk.
"""

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Callable, Generator, Optional

import numpy as np

try:
    import sounddevice as sd
except OSError:
    sd = None  # Audio not available (e.g., in Docker without audio)

from vosk import KaldiRecognizer, Model, SetLogLevel

from .dictionary import STTDictionary

logger = logging.getLogger(__name__)


class VoskSTT:
    """
    Vosk-based Speech-to-Text engine.

    Supports both batch and streaming recognition.
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        chunk_size: int = 8000,
        dictionary: Optional[STTDictionary] = None,
    ):
        """
        Initialize Vosk STT.

        Args:
            model_path: Path to Vosk model directory
            sample_rate: Audio sample rate (default: 16000)
            chunk_size: Audio chunk size for processing
            dictionary: Optional STT correction dictionary
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.dictionary = dictionary
        self._audio_queue: queue.Queue = queue.Queue()
        self._is_recording = False

        # Suppress Vosk debug logs
        SetLogLevel(-1)

        # Load model
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Vosk model not found: {model_path}")

        logger.info(f"Loading Vosk model from: {model_path}")
        self.model = Model(str(model_path))
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.recognizer.SetWords(True)

        logger.info("Vosk STT initialized")

    def recognize(self, audio_data: np.ndarray) -> str:
        """
        Recognize speech from audio data.

        Args:
            audio_data: Audio data as numpy array (int16)

        Returns:
            Recognized text
        """
        # Create fresh recognizer for each recognition
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)

        # Ensure correct format (int16)
        if audio_data.dtype != np.int16:
            if np.issubdtype(audio_data.dtype, np.floating):
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)

        logger.debug(f"STT input: {len(audio_data)} samples, dtype={audio_data.dtype}, max={np.max(np.abs(audio_data))}")

        # Process audio in chunks (Vosk works better this way)
        chunk_size = 4000
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            recognizer.AcceptWaveform(chunk.tobytes())

        result = json.loads(recognizer.FinalResult())
        text = result.get("text", "")

        # Apply dictionary corrections if available
        if self.dictionary and text:
            corrected = self.dictionary.correct(text)
            if corrected != text:
                logger.debug(f"STT corrected: '{text}' -> '{corrected}'")
                text = corrected

        logger.debug(f"STT result: '{text}'")

        return text

    def recognize_file(self, audio_path: str) -> str:
        """
        Recognize speech from an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Recognized text
        """
        import soundfile as sf

        audio_data, sr = sf.read(audio_path, dtype="int16")

        # Resample if necessary
        if sr != self.sample_rate:
            import scipy.signal

            audio_data = scipy.signal.resample(
                audio_data, int(len(audio_data) * self.sample_rate / sr)
            ).astype(np.int16)

        return self.recognize(audio_data)

    def stream_recognize(
        self,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
        timeout: float = 10.0,
    ) -> str:
        """
        Stream recognition from microphone.

        Args:
            on_partial: Callback for partial recognition results
            on_final: Callback for final recognition result
            timeout: Maximum recording time in seconds

        Returns:
            Final recognized text
        """
        if sd is None:
            raise RuntimeError("sounddevice not available")

        self._audio_queue = queue.Queue()
        self._is_recording = True
        final_text = ""

        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio status: {status}")
            self._audio_queue.put(indata.copy())

        # Reset recognizer
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.chunk_size,
                callback=audio_callback,
            ):
                logger.info("Recording started...")
                elapsed = 0.0
                check_interval = 0.1

                while self._is_recording and elapsed < timeout:
                    try:
                        data = self._audio_queue.get(timeout=check_interval)
                        elapsed += check_interval

                        if self.recognizer.AcceptWaveform(data.tobytes()):
                            result = json.loads(self.recognizer.Result())
                            text = result.get("text", "")
                            if text and self.dictionary:
                                text = self.dictionary.correct(text)
                            if text and on_final:
                                on_final(text)
                            final_text = text
                        else:
                            partial = json.loads(self.recognizer.PartialResult())
                            partial_text = partial.get("partial", "")
                            if partial_text and on_partial:
                                on_partial(partial_text)

                    except queue.Empty:
                        continue

                # Get final result
                result = json.loads(self.recognizer.FinalResult())
                final_text = result.get("text", final_text)
                if final_text and self.dictionary:
                    final_text = self.dictionary.correct(final_text)

        except Exception as e:
            logger.error(f"Recording error: {e}")
            raise

        finally:
            self._is_recording = False
            logger.info("Recording stopped")

        return final_text

    def stop_recording(self):
        """Stop the current recording session."""
        self._is_recording = False

    def stream_generator(self) -> Generator[dict, None, None]:
        """
        Generator for streaming recognition results.

        Yields:
            Dict with 'partial' or 'final' text
        """
        if sd is None:
            raise RuntimeError("sounddevice not available")

        self._audio_queue = queue.Queue()
        self._is_recording = True

        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio status: {status}")
            self._audio_queue.put(indata.copy())

        # Reset recognizer
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.chunk_size,
                callback=audio_callback,
            ):
                while self._is_recording:
                    try:
                        data = self._audio_queue.get(timeout=0.1)

                        if self.recognizer.AcceptWaveform(data.tobytes()):
                            result = json.loads(self.recognizer.Result())
                            text = result.get("text", "")
                            if text and self.dictionary:
                                text = self.dictionary.correct(text)
                            if text:
                                yield {"type": "final", "text": text}
                        else:
                            partial = json.loads(self.recognizer.PartialResult())
                            partial_text = partial.get("partial", "")
                            if partial_text:
                                yield {"type": "partial", "text": partial_text}

                    except queue.Empty:
                        continue

                # Final result
                result = json.loads(self.recognizer.FinalResult())
                text = result.get("text", "")
                if text and self.dictionary:
                    text = self.dictionary.correct(text)
                if text:
                    yield {"type": "final", "text": text}

        finally:
            self._is_recording = False


if __name__ == "__main__":
    # Test STT
    logging.basicConfig(level=logging.INFO)

    model_path = "models/vosk/vosk-model-small-ja-0.22"
    stt = VoskSTT(model_path)

    print("Recording for 5 seconds...")
    text = stt.stream_recognize(
        on_partial=lambda t: print(f"[partial] {t}"),
        on_final=lambda t: print(f"[FINAL] {t}"),
        timeout=5.0,
    )
    print(f"\nFinal result: {text}")
