"""
Unit tests for QwenTTS module.
Tests dual-mode operation, voice clone prompt caching, thread safety,
pronunciation dictionary, and language consistency.
"""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.tts.qwen_tts import MODEL_MAP, QwenTTS


# ---------------------------------------------------------------------------
# Model mapping tests
# ---------------------------------------------------------------------------
class TestModelMapping:
    """Test that the correct model is selected based on mode."""

    def test_custom_voice_model(self):
        assert MODEL_MAP["custom_voice"] == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    def test_voice_clone_model(self):
        assert MODEL_MAP["voice_clone"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    def test_voice_design_model(self):
        assert MODEL_MAP["voice_design"] == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------
class TestQwenTTSInit:
    """Test QwenTTS initialization with different modes."""

    @patch("src.utils.device.detect_device")
    def test_init_custom_voice_mode(self, mock_detect, mock_device_config):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="custom_voice")

        assert tts.mode == "custom_voice"
        assert tts.model_name == MODEL_MAP["custom_voice"]
        assert tts._voice_clone_prompt is None
        assert tts._clone_language == "Japanese"

    @patch("src.utils.device.detect_device")
    def test_init_voice_clone_mode(self, mock_detect, mock_device_config):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")

        assert tts.mode == "voice_clone"
        assert tts.model_name == MODEL_MAP["voice_clone"]

    @patch("src.utils.device.detect_device")
    def test_init_explicit_model_overrides_auto(self, mock_detect, mock_device_config):
        mock_detect.return_value = mock_device_config
        custom_model = "my-org/my-custom-model"
        tts = QwenTTS(model_name=custom_model, mode="voice_clone")

        assert tts.model_name == custom_model

    @patch("src.utils.device.detect_device")
    def test_init_unknown_mode_falls_back(self, mock_detect, mock_device_config):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="nonexistent_mode")

        assert tts.model_name == MODEL_MAP["custom_voice"]

    @patch("src.utils.device.detect_device")
    def test_init_has_thread_lock(self, mock_detect, mock_device_config):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")

        assert hasattr(tts, "_prompt_lock")
        assert isinstance(tts._prompt_lock, type(threading.RLock()))

    @patch("src.utils.device.detect_device")
    def test_init_has_model_lock(self, mock_detect, mock_device_config):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")

        assert hasattr(tts, "_model_lock")
        assert isinstance(tts._model_lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# Pronunciation dictionary tests
# ---------------------------------------------------------------------------
class TestPronunciationDict:
    """Test pronunciation dictionary loading and application."""

    @patch("src.utils.device.detect_device")
    def test_load_pronunciation_dict(self, mock_detect, mock_device_config, pronunciation_dict_path):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(pronunciation_dict_path=pronunciation_dict_path)

        assert len(tts.pronunciation_dict.get("terms", [])) == 2

    @patch("src.utils.device.detect_device")
    def test_preprocess_term_replacement(self, mock_detect, mock_device_config, pronunciation_dict_path):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(pronunciation_dict_path=pronunciation_dict_path)

        result = tts._preprocess_text("WellAIへようこそ")
        assert "ウェルアイ" in result
        assert "WellAI" not in result

    @patch("src.utils.device.detect_device")
    def test_preprocess_pattern_replacement(self, mock_detect, mock_device_config, pronunciation_dict_path):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(pronunciation_dict_path=pronunciation_dict_path)

        result = tts._preprocess_text("料金は500円です")
        assert "500えん" in result

    @patch("src.utils.device.detect_device")
    def test_preprocess_no_dict(self, mock_detect, mock_device_config):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS()

        text = "WellAI 500円"
        assert tts._preprocess_text(text) == text

    @patch("src.utils.device.detect_device")
    def test_load_nonexistent_dict(self, mock_detect, mock_device_config, tmp_path):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(pronunciation_dict_path=str(tmp_path / "nonexistent.yaml"))

        assert tts.pronunciation_dict == {}

    @patch("src.utils.device.detect_device")
    def test_load_empty_yaml_dict(self, mock_detect, mock_device_config, tmp_path):
        """Empty YAML file should not crash (Codex finding #4)."""
        mock_detect.return_value = mock_device_config
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")
        tts = QwenTTS(pronunciation_dict_path=str(empty_file))

        assert tts.pronunciation_dict == {}

    @patch("src.utils.device.detect_device")
    def test_load_non_dict_yaml(self, mock_detect, mock_device_config, tmp_path):
        """YAML that parses to a list (not dict) should not crash."""
        mock_detect.return_value = mock_device_config
        list_file = tmp_path / "list.yaml"
        list_file.write_text("- item1\n- item2\n", encoding="utf-8")
        tts = QwenTTS(pronunciation_dict_path=str(list_file))

        assert tts.pronunciation_dict == {}


# ---------------------------------------------------------------------------
# Custom voice synthesis tests
# ---------------------------------------------------------------------------
class TestSynthesizeCustomVoice:
    """Test custom voice synthesis."""

    @patch("src.utils.device.detect_device")
    def test_synthesize_calls_generate_custom_voice(
        self, mock_detect, mock_device_config, mock_tts_model
    ):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="custom_voice")
        tts.model = mock_tts_model

        audio, sr = tts.synthesize("テスト", speaker="ono_anna", language="Japanese")

        mock_tts_model.generate_custom_voice.assert_called_once_with(
            text="テスト",
            language="Japanese",
            speaker="ono_anna",
            instruct="",
        )
        assert sr == 24000
        assert isinstance(audio, np.ndarray)

    @patch("src.utils.device.detect_device")
    def test_synthesize_always_passes_language(
        self, mock_detect, mock_device_config, mock_tts_model
    ):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="custom_voice")
        tts.model = mock_tts_model

        tts.synthesize("テスト", language="Japanese")
        call_args = mock_tts_model.generate_custom_voice.call_args
        assert call_args.kwargs["language"] == "Japanese"

    @patch("src.utils.device.detect_device")
    def test_synthesize_applies_pronunciation_dict(
        self, mock_detect, mock_device_config, mock_tts_model, pronunciation_dict_path
    ):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(pronunciation_dict_path=pronunciation_dict_path)
        tts.model = mock_tts_model

        tts.synthesize("WellAIです", language="Japanese")
        call_args = mock_tts_model.generate_custom_voice.call_args
        assert "ウェルアイ" in call_args.kwargs["text"]


# ---------------------------------------------------------------------------
# Voice clone synthesis tests
# ---------------------------------------------------------------------------
class TestSynthesizeWithClone:
    """Test voice cloning synthesis with prompt caching."""

    @patch("src.utils.device.detect_device")
    def test_clone_uses_cached_prompt(
        self, mock_detect, mock_device_config, mock_tts_model
    ):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model
        tts._voice_clone_prompt = {"cached": True}
        tts._clone_language = "Japanese"

        audio, sr = tts.synthesize_with_clone("テスト")

        mock_tts_model.generate_voice_clone.assert_called_once()
        call_args = mock_tts_model.generate_voice_clone.call_args
        assert call_args.kwargs["voice_clone_prompt"] == {"cached": True}
        assert call_args.kwargs["language"] == "Japanese"

    @patch("src.utils.device.detect_device")
    def test_clone_uses_cached_language_not_parameter(
        self, mock_detect, mock_device_config, mock_tts_model
    ):
        """CRITICAL FIX: cached language must be used, not the parameter."""
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model
        tts._voice_clone_prompt = {"cached": True}
        tts._clone_language = "Japanese"

        # Even if caller passes English, cached Japanese should be used
        tts.synthesize_with_clone("テスト", language="English")

        call_args = mock_tts_model.generate_voice_clone.call_args
        assert call_args.kwargs["language"] == "Japanese"

    @patch("src.utils.device.detect_device")
    def test_clone_without_cache_uses_ref_audio(
        self, mock_detect, mock_device_config, mock_tts_model, ref_audio_file
    ):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model

        tts.synthesize_with_clone(
            "テスト",
            ref_audio_path=ref_audio_file,
            ref_text="リファレンス",
            language="Japanese",
        )

        call_args = mock_tts_model.generate_voice_clone.call_args
        assert call_args.kwargs["ref_audio"] == ref_audio_file
        assert call_args.kwargs["language"] == "Japanese"

    @patch("src.utils.device.detect_device")
    def test_clone_without_cache_or_ref_raises(
        self, mock_detect, mock_device_config, mock_tts_model
    ):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model

        with pytest.raises(ValueError, match="No cached voice clone prompt"):
            tts.synthesize_with_clone("テスト")

    @patch("src.utils.device.detect_device")
    def test_clone_with_nonexistent_ref_raises(
        self, mock_detect, mock_device_config, mock_tts_model
    ):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model

        with pytest.raises(FileNotFoundError):
            tts.synthesize_with_clone(
                "テスト",
                ref_audio_path="/nonexistent/audio.wav",
            )


# ---------------------------------------------------------------------------
# Voice clone prompt caching tests
# ---------------------------------------------------------------------------
class TestPrepareClone:
    """Test voice clone prompt pre-computation and caching."""

    @patch("src.utils.device.detect_device")
    def test_prepare_clone_caches_prompt(
        self, mock_detect, mock_device_config, mock_tts_model, ref_audio_file
    ):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model

        tts.prepare_clone(ref_audio_file, "テスト", language="Japanese")

        assert tts._voice_clone_prompt is not None
        assert tts._clone_language == "Japanese"
        mock_tts_model.create_voice_clone_prompt.assert_called_once()

    @patch("src.utils.device.detect_device")
    def test_prepare_clone_nonexistent_file(
        self, mock_detect, mock_device_config, mock_tts_model
    ):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model

        tts.prepare_clone("/nonexistent/audio.wav", "テスト")

        assert tts._voice_clone_prompt is None

    @patch("src.utils.device.detect_device")
    def test_prepare_clone_short_audio_rejected(
        self, mock_detect, mock_device_config, mock_tts_model, short_audio_file
    ):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model

        tts.prepare_clone(short_audio_file, "テスト")

        # Short audio (<1s) should not cache prompt
        assert tts._voice_clone_prompt is None

    @patch("src.utils.device.detect_device")
    def test_update_reference_audio(
        self, mock_detect, mock_device_config, mock_tts_model, ref_audio_file
    ):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model

        # Set initial prompt
        tts._voice_clone_prompt = {"old": True}

        # Update with new audio
        tts.update_reference_audio(ref_audio_file, "新しいテスト", language="Japanese")

        # Should have new prompt
        assert tts._voice_clone_prompt is not None
        assert tts._clone_language == "Japanese"


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------
class TestThreadSafety:
    """Test concurrent access to voice clone prompt cache."""

    @patch("src.utils.device.detect_device")
    def test_concurrent_synthesize_with_cached_prompt(
        self, mock_detect, mock_device_config, mock_tts_model
    ):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model
        tts._voice_clone_prompt = {"cached": True}
        tts._clone_language = "Japanese"

        results = []
        errors = []

        def do_synthesize():
            try:
                audio, sr = tts.synthesize_with_clone("並列テスト")
                results.append((audio, sr))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_synthesize) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Errors in concurrent access: {errors}"
        assert len(results) == 5

    @patch("src.utils.device.detect_device")
    def test_concurrent_update_and_synthesize(
        self, mock_detect, mock_device_config, mock_tts_model, ref_audio_file
    ):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model
        tts._voice_clone_prompt = {"cached": True}
        tts._clone_language = "Japanese"

        errors = []

        def do_synthesize():
            try:
                for _ in range(3):
                    tts.synthesize_with_clone("並列テスト")
            except Exception as e:
                errors.append(e)

        def do_update():
            try:
                tts.update_reference_audio(ref_audio_file, "更新テスト")
            except Exception as e:
                errors.append(e)

        synth_threads = [threading.Thread(target=do_synthesize) for _ in range(3)]
        update_thread = threading.Thread(target=do_update)

        for t in synth_threads:
            t.start()
        update_thread.start()

        for t in synth_threads:
            t.join(timeout=5)
        update_thread.join(timeout=5)

        assert len(errors) == 0, f"Errors in concurrent update: {errors}"


# ---------------------------------------------------------------------------
# switch_mode tests
# ---------------------------------------------------------------------------
class TestSwitchMode:
    """Test runtime mode switching via switch_mode()."""

    @patch("src.utils.device.detect_device")
    def test_switch_mode_changes_mode_and_model(self, mock_detect, mock_device_config):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="custom_voice")

        assert tts.mode == "custom_voice"
        assert tts.model_name == MODEL_MAP["custom_voice"]

        tts.switch_mode("voice_clone")

        assert tts.mode == "voice_clone"
        assert tts.model_name == MODEL_MAP["voice_clone"]

    @patch("src.utils.device.detect_device")
    def test_switch_mode_unloads_model(self, mock_detect, mock_device_config, mock_tts_model):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="custom_voice")
        tts.model = mock_tts_model  # simulate loaded model

        assert tts.model is not None

        tts.switch_mode("voice_clone")

        assert tts.model is None

    @patch("src.utils.device.detect_device")
    def test_switch_mode_clears_clone_prompt(self, mock_detect, mock_device_config):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts._voice_clone_prompt = {"cached": True}

        tts.switch_mode("custom_voice")

        assert tts._voice_clone_prompt is None

    @patch("src.utils.device.detect_device")
    def test_switch_mode_thread_safe(self, mock_detect, mock_device_config, mock_tts_model):
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="custom_voice")
        tts.model = mock_tts_model
        tts._voice_clone_prompt = {"cached": True}

        errors = []

        def do_switch(target_mode):
            try:
                tts.switch_mode(target_mode)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            mode = "voice_clone" if i % 2 == 0 else "custom_voice"
            threads.append(threading.Thread(target=do_switch, args=(mode,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Errors in concurrent switch_mode: {errors}"
        # Final mode should be one of the valid modes
        assert tts.mode in ("custom_voice", "voice_clone")

    @patch("src.utils.device.detect_device")
    def test_switch_mode_same_mode_skips_model_unload(
        self, mock_detect, mock_device_config, mock_tts_model
    ):
        """Switching to the same mode should NOT unload the model."""
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts.model = mock_tts_model  # simulate loaded model

        tts.switch_mode("voice_clone")

        # Model should still be loaded (not unloaded)
        assert tts.model is mock_tts_model
        assert tts.mode == "voice_clone"

    @patch("src.utils.device.detect_device")
    def test_switch_mode_same_mode_clears_prompt(
        self, mock_detect, mock_device_config
    ):
        """Even when staying in the same mode, prompt cache should be cleared."""
        mock_detect.return_value = mock_device_config
        tts = QwenTTS(mode="voice_clone")
        tts._voice_clone_prompt = {"cached": True}

        tts.switch_mode("voice_clone")

        assert tts._voice_clone_prompt is None


# ---------------------------------------------------------------------------
# Speaker listing
# ---------------------------------------------------------------------------
class TestSpeakerListing:
    """Test speaker listing utility."""

    def test_list_speakers_returns_copy(self):
        speakers = QwenTTS.list_speakers()
        assert "ono_anna" in speakers
        # Ensure it's a copy
        speakers["test"] = "test"
        assert "test" not in QwenTTS.SPEAKERS

    def test_speakers_include_japanese(self):
        speakers = QwenTTS.list_speakers()
        assert "ono_anna" in speakers
        assert "Japanese" in speakers["ono_anna"]
