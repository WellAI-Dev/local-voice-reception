"""
Integration tests for the TTS pipeline.
Tests the full flow from config → TTS init → synthesis routing.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.app import VoiceReceptionApp
from src.tts.qwen_tts import QwenTTS


# ---------------------------------------------------------------------------
# Full pipeline: config → TTS → synthesis
# ---------------------------------------------------------------------------
class TestTTSPipelineIntegration:
    """Integration test: config drives TTS mode, which drives synthesis method."""

    @patch("src.utils.device.detect_device")
    def test_custom_voice_pipeline(self, mock_detect, mock_device_config, mock_tts_model):
        """Config mode=custom_voice → synthesize() → generate_custom_voice()."""
        mock_detect.return_value = mock_device_config

        config = {
            "tts": {
                "mode": "custom_voice",
                "device": "cpu",
                "custom_voice": {"speaker": "ono_anna", "language": "Japanese"},
                "voice_clone": {"ref_audio": "", "ref_text": ""},
            }
        }
        app = VoiceReceptionApp(config)
        tts = QwenTTS(mode="custom_voice")
        tts.model = mock_tts_model
        app.tts = tts

        audio, sr = app._synthesize_speech("テスト音声")

        mock_tts_model.generate_custom_voice.assert_called_once()
        call_args = mock_tts_model.generate_custom_voice.call_args
        assert call_args.kwargs["language"] == "Japanese"
        assert call_args.kwargs["speaker"] == "ono_anna"
        assert sr == 24000

    @patch("src.utils.device.detect_device")
    def test_voice_clone_pipeline_with_cached_prompt(
        self, mock_detect, mock_device_config, mock_tts_model
    ):
        """Config mode=voice_clone + cached prompt → synthesize_with_clone() → generate_voice_clone()."""
        mock_detect.return_value = mock_device_config

        config = {
            "tts": {
                "mode": "voice_clone",
                "device": "cpu",
                "custom_voice": {"speaker": "ono_anna", "language": "Japanese"},
                "voice_clone": {"ref_audio": "", "ref_text": ""},
            }
        }
        app = VoiceReceptionApp(config)
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model
        tts._voice_clone_prompt = {"cached": True}
        tts._clone_language = "Japanese"
        app.tts = tts

        audio, sr = app._synthesize_speech("テスト音声")

        mock_tts_model.generate_voice_clone.assert_called_once()
        call_args = mock_tts_model.generate_voice_clone.call_args
        assert call_args.kwargs["language"] == "Japanese"
        assert call_args.kwargs["voice_clone_prompt"] == {"cached": True}

    @patch("src.utils.device.detect_device")
    def test_voice_clone_fallback_to_custom_voice(
        self, mock_detect, mock_device_config, mock_tts_model
    ):
        """Config mode=voice_clone but no cached prompt → falls back to synthesize()."""
        mock_detect.return_value = mock_device_config

        config = {
            "tts": {
                "mode": "voice_clone",
                "device": "cpu",
                "custom_voice": {"speaker": "ono_anna", "language": "Japanese"},
                "voice_clone": {"ref_audio": "", "ref_text": ""},
            }
        }
        app = VoiceReceptionApp(config)
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model
        # No cached prompt
        app.tts = tts

        audio, sr = app._synthesize_speech("テスト音声")

        # Should have fallen back to custom_voice
        mock_tts_model.generate_custom_voice.assert_called_once()
        mock_tts_model.generate_voice_clone.assert_not_called()


# ---------------------------------------------------------------------------
# Voice clone prompt lifecycle
# ---------------------------------------------------------------------------
class TestVoiceClonePromptLifecycle:
    """Test the full lifecycle: prepare → synthesize → update → synthesize."""

    @patch("src.utils.device.detect_device")
    def test_prepare_then_synthesize_uses_cache(
        self, mock_detect, mock_device_config, mock_tts_model, ref_audio_file
    ):
        mock_detect.return_value = mock_device_config

        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model

        # Step 1: prepare_clone
        tts.prepare_clone(ref_audio_file, "参照テキスト", language="Japanese")
        assert tts._voice_clone_prompt is not None
        first_prompt = tts._voice_clone_prompt

        # Step 2: synthesize uses cache
        tts.synthesize_with_clone("こんにちは")
        call_args = mock_tts_model.generate_voice_clone.call_args
        assert call_args.kwargs["voice_clone_prompt"] == first_prompt
        assert call_args.kwargs["language"] == "Japanese"

    @patch("src.utils.device.detect_device")
    def test_update_replaces_cache(
        self, mock_detect, mock_device_config, mock_tts_model, ref_audio_file
    ):
        mock_detect.return_value = mock_device_config

        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model

        # Initial setup
        tts.prepare_clone(ref_audio_file, "初期テキスト", language="Japanese")
        first_prompt = tts._voice_clone_prompt

        # Update with "new" audio (same file but new call)
        mock_tts_model.create_voice_clone_prompt.return_value = {"new_cached": True}
        tts.update_reference_audio(ref_audio_file, "新テキスト", language="Japanese")

        assert tts._voice_clone_prompt == {"new_cached": True}
        assert tts._voice_clone_prompt != first_prompt

    @patch("src.utils.device.detect_device")
    def test_multiple_synthesize_calls_use_same_cache(
        self, mock_detect, mock_device_config, mock_tts_model, ref_audio_file
    ):
        """Verify that multiple calls reuse the same cached prompt for consistency."""
        mock_detect.return_value = mock_device_config

        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model

        tts.prepare_clone(ref_audio_file, "テスト", language="Japanese")

        # Call synthesize 3 times
        for _ in range(3):
            tts.synthesize_with_clone("テスト")

        # create_voice_clone_prompt should only be called once (during prepare)
        assert mock_tts_model.create_voice_clone_prompt.call_count == 1
        # generate_voice_clone should be called 3 times
        assert mock_tts_model.generate_voice_clone.call_count == 3


# ---------------------------------------------------------------------------
# Pronunciation + TTS integration
# ---------------------------------------------------------------------------
class TestPronunciationTTSIntegration:
    """Test pronunciation dictionary applied before TTS generation."""

    @patch("src.utils.device.detect_device")
    def test_pronunciation_applied_in_custom_voice(
        self, mock_detect, mock_device_config, mock_tts_model, pronunciation_dict_path
    ):
        mock_detect.return_value = mock_device_config

        tts = QwenTTS(mode="custom_voice", pronunciation_dict_path=pronunciation_dict_path)
        tts.model = mock_tts_model

        tts.synthesize("Cor.IncのAPIは500円です", language="Japanese")

        call_args = mock_tts_model.generate_custom_voice.call_args
        text = call_args.kwargs["text"]
        assert "コア インク" in text
        assert "エーピーアイ" in text
        assert "500えん" in text

    @patch("src.utils.device.detect_device")
    def test_pronunciation_applied_in_voice_clone(
        self, mock_detect, mock_device_config, mock_tts_model, pronunciation_dict_path
    ):
        mock_detect.return_value = mock_device_config

        tts = QwenTTS(mode="voice_clone", pronunciation_dict_path=pronunciation_dict_path)
        tts.model = mock_tts_model
        tts._voice_clone_prompt = {"cached": True}
        tts._clone_language = "Japanese"

        tts.synthesize_with_clone("Cor.Incへようこそ")

        call_args = mock_tts_model.generate_voice_clone.call_args
        text = call_args.kwargs["text"]
        assert "コア インク" in text
