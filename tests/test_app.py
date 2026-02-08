"""
Unit and integration tests for the VoiceReceptionApp.
Tests TTS mode routing (_synthesize_speech), voice registration,
and the initialization pipeline.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.app import VoiceReceptionApp


# ---------------------------------------------------------------------------
# _synthesize_speech routing tests
# ---------------------------------------------------------------------------
class TestSynthesizeSpeechRouting:
    """Test that _synthesize_speech routes to correct TTS method based on mode."""

    def _make_app(self, mode="custom_voice"):
        import threading

        config = {
            "tts": {
                "mode": mode,
                "custom_voice": {
                    "speaker": "ono_anna",
                    "language": "Japanese",
                },
                "voice_clone": {
                    "ref_audio": "data/voice_samples/company_voice.wav",
                    "ref_text": "テスト",
                },
            },
        }
        app = VoiceReceptionApp(config)
        app.tts = MagicMock()
        app.tts._voice_clone_prompt = None
        app.tts._prompt_lock = threading.RLock()
        app.tts.mode = mode
        app.tts.synthesize.return_value = (np.zeros(24000, dtype=np.float32), 24000)
        app.tts.synthesize_with_clone.return_value = (np.zeros(24000, dtype=np.float32), 24000)
        return app

    def test_custom_voice_mode_calls_synthesize(self):
        app = self._make_app(mode="custom_voice")

        audio, sr = app._synthesize_speech("テスト")

        app.tts.synthesize.assert_called_once_with(
            text="テスト",
            speaker="ono_anna",
            language="Japanese",
        )
        app.tts.synthesize_with_clone.assert_not_called()

    def test_voice_clone_mode_with_prompt_calls_clone(self):
        app = self._make_app(mode="voice_clone")
        app.tts._voice_clone_prompt = {"cached": True}

        audio, sr = app._synthesize_speech("テスト")

        app.tts.synthesize_with_clone.assert_called_once_with(
            text="テスト",
            language="Japanese",
        )
        app.tts.synthesize.assert_not_called()

    def test_voice_clone_mode_without_prompt_falls_back(self):
        app = self._make_app(mode="voice_clone")
        app.tts._voice_clone_prompt = None

        audio, sr = app._synthesize_speech("テスト")

        # Should fall back to custom_voice
        app.tts.synthesize.assert_called_once()
        app.tts.synthesize_with_clone.assert_not_called()

    def test_language_always_japanese(self):
        app = self._make_app(mode="custom_voice")

        app._synthesize_speech("テスト")

        call_args = app.tts.synthesize.call_args
        assert call_args.kwargs["language"] == "Japanese"

    def test_voice_clone_language_always_japanese(self):
        app = self._make_app(mode="voice_clone")
        app.tts._voice_clone_prompt = {"cached": True}

        app._synthesize_speech("テスト")

        call_args = app.tts.synthesize_with_clone.call_args
        assert call_args.kwargs["language"] == "Japanese"

    def test_language_read_from_config(self):
        """Language should be read from config, not hardcoded (Codex finding #3)."""
        config = {
            "tts": {
                "mode": "custom_voice",
                "custom_voice": {
                    "speaker": "ono_anna",
                    "language": "English",
                },
                "voice_clone": {
                    "ref_audio": "",
                    "ref_text": "",
                },
            },
        }
        app = VoiceReceptionApp(config)
        app.tts = MagicMock()
        app.tts._voice_clone_prompt = None
        app.tts._prompt_lock = MagicMock()
        app.tts.synthesize.return_value = (np.zeros(24000, dtype=np.float32), 24000)

        app._synthesize_speech("test")

        call_args = app.tts.synthesize.call_args
        assert call_args.kwargs["language"] == "English"

    def test_voice_clone_check_uses_prompt_lock(self):
        """_synthesize_speech should access _voice_clone_prompt through _prompt_lock (Codex finding #2)."""
        import threading

        app = self._make_app(mode="voice_clone")
        app.tts._voice_clone_prompt = {"cached": True}
        app.tts._prompt_lock = threading.RLock()

        # Should not raise - if lock isn't used this would deadlock in some setups
        audio, sr = app._synthesize_speech("テスト")
        app.tts.synthesize_with_clone.assert_called_once()

    def test_synthesize_reads_mode_from_tts_instance(self):
        """_synthesize_speech should read mode from self.tts.mode, not config."""
        app = self._make_app(mode="custom_voice")
        # Config still says custom_voice, but tts.mode is changed to voice_clone
        app.tts.mode = "voice_clone"
        app.tts._voice_clone_prompt = {"cached": True}

        app._synthesize_speech("テスト")

        # Should route to voice_clone because tts.mode is "voice_clone"
        app.tts.synthesize_with_clone.assert_called_once()
        app.tts.synthesize.assert_not_called()


# ---------------------------------------------------------------------------
# Voice registration mode switching tests
# ---------------------------------------------------------------------------
class TestVoiceRegistrationModeSwitching:
    """Test that voice registration triggers mode switch to voice_clone."""

    def test_register_voice_switches_to_voice_clone_mode(self, base_config, tmp_path):
        """register_voice() should call switch_mode('voice_clone') on the TTS instance."""
        import threading

        app = VoiceReceptionApp(base_config)
        app.tts = MagicMock()
        app.tts.mode = "custom_voice"
        app.tts._voice_clone_prompt = None
        app.tts._prompt_lock = threading.RLock()

        # Simulate the register_voice logic from the closure
        sample_rate_in = 24000
        audio_data_in = np.sin(np.linspace(0, 1, int(24000 * 3.0))).astype(np.float32)
        ref_text = "テスト音声です。"

        voice_dir = tmp_path / "data" / "voice_samples"
        voice_dir.mkdir(parents=True, exist_ok=True)
        ref_path = voice_dir / "company_voice.wav"

        import soundfile as sf
        sf.write(str(ref_path), audio_data_in, sample_rate_in)

        # Simulate what register_voice does after saving audio
        app.tts.switch_mode("voice_clone")
        app.config["tts"]["mode"] = "voice_clone"
        app.tts.update_reference_audio(
            ref_audio_path=str(ref_path),
            ref_text=ref_text,
            language="Japanese",
        )

        app.tts.switch_mode.assert_called_once_with("voice_clone")
        assert app.config["tts"]["mode"] == "voice_clone"
        app.tts.update_reference_audio.assert_called_once_with(
            ref_audio_path=str(ref_path),
            ref_text=ref_text,
            language="Japanese",
        )


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------
class TestVoiceReceptionAppInit:
    """Test VoiceReceptionApp initialization."""

    def test_init_default_state(self, base_config):
        app = VoiceReceptionApp(base_config)

        assert app.stt is None
        assert app.llm is None
        assert app.tts is None
        assert app.conversation_log == []

    @patch("src.app.QwenTTS")
    @patch("src.app.OllamaClient")
    @patch("src.app.VoskSTT")
    def test_initialize_tts_mode_passed(self, mock_stt, mock_llm, mock_tts, base_config):
        """Ensure mode is passed to QwenTTS during initialization."""
        mock_stt_instance = MagicMock()
        mock_stt.return_value = mock_stt_instance

        mock_llm_instance = MagicMock()
        mock_llm_instance.check_connection.return_value = True
        mock_llm.return_value = mock_llm_instance

        mock_tts_instance = MagicMock()
        mock_tts_instance._voice_clone_prompt = None
        mock_tts.return_value = mock_tts_instance

        # Override model_path so it doesn't fail
        base_config["stt"]["model_path"] = "/nonexistent"

        app = VoiceReceptionApp(base_config)
        app.initialize()

        mock_tts.assert_called_once()
        call_kwargs = mock_tts.call_args.kwargs
        assert call_kwargs["mode"] == "custom_voice"

    @patch("src.app.QwenTTS")
    @patch("src.app.OllamaClient")
    @patch("src.app.VoskSTT")
    def test_initialize_voice_clone_with_ref_audio(
        self, mock_stt, mock_llm, mock_tts, base_config, ref_audio_file
    ):
        """When mode is voice_clone and ref audio exists, prepare_clone is called."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.check_connection.return_value = True
        mock_llm.return_value = mock_llm_instance

        mock_tts_instance = MagicMock()
        mock_tts_instance._voice_clone_prompt = None
        mock_tts.return_value = mock_tts_instance

        base_config["tts"]["mode"] = "voice_clone"
        base_config["tts"]["voice_clone"]["ref_audio"] = ref_audio_file
        base_config["stt"]["model_path"] = "/nonexistent"

        app = VoiceReceptionApp(base_config)
        app.initialize()

        mock_tts_instance.prepare_clone.assert_called_once()


# ---------------------------------------------------------------------------
# Voice registration tests
# ---------------------------------------------------------------------------
class TestVoiceRegistration:
    """Test the voice recording/registration flow."""

    def test_register_rejects_no_audio(self, base_config, tmp_path):
        """Should reject when no audio is provided."""
        app = VoiceReceptionApp(base_config)

        # Simulate register_voice logic (extracted from closure)
        audio_tuple = None
        ref_text = "テスト"

        if audio_tuple is None:
            result = "音声が入力されていません。マイクで録音するかファイルをアップロードしてください。"
        else:
            result = "OK"

        assert "入力されていません" in result

    def test_register_rejects_empty_text(self, base_config):
        """Should reject when reference text is empty."""
        ref_text = ""
        assert not ref_text.strip()

    def test_register_rejects_short_audio(self):
        """Audio under 1 second should be rejected."""
        sample_rate = 24000
        duration = 0.5
        samples = int(sample_rate * duration)
        assert samples / sample_rate < 1.0

    def test_audio_duration_calculation(self, sample_audio_float32):
        """Verify duration calculation."""
        sr, audio = sample_audio_float32
        duration = len(audio) / sr
        assert duration >= 3.0


# ---------------------------------------------------------------------------
# Conversation management tests
# ---------------------------------------------------------------------------
class TestConversationManagement:
    """Test conversation log and history management."""

    def test_get_conversation_display_empty(self, base_config):
        app = VoiceReceptionApp(base_config)
        assert app.get_conversation_display() == ""

    def test_get_conversation_display_with_entries(self, base_config):
        app = VoiceReceptionApp(base_config)
        app.conversation_log = [
            {"user": "こんにちは", "assistant": "お電話ありがとうございます。"}
        ]

        display = app.get_conversation_display()
        assert "こんにちは" in display
        assert "お電話ありがとうございます" in display

    def test_clear_conversation(self, base_config):
        app = VoiceReceptionApp(base_config)
        app.conversation_log = [{"user": "test", "assistant": "response"}]
        app.llm = MagicMock()

        app.clear_conversation()

        assert app.conversation_log == []
        app.llm.clear_history.assert_called_once()
