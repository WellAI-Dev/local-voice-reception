"""
Unit tests for STTDictionary module.
Tests loading/saving, text correction, pattern matching,
CRUD operations, and VoskSTT integration.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from src.stt.dictionary import STTDictionary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def stt_dict_path(tmp_path):
    """Create a temporary STT dictionary YAML file."""
    dict_content = {
        "corrections": [
            {"wrong": "みらい", "correct": "MIRAI", "note": "製品名"},
            {"wrong": "えーぴーあい", "correct": "API", "note": "技術用語"},
            {"wrong": "うぇるあい", "correct": "WellAI", "note": "会社名"},
        ],
        "patterns": [
            {"pattern": "おでんわ", "replacement": "お電話"},
            {"pattern": r"(\d+)えん", "replacement": r"\1円"},
        ],
    }
    dict_path = tmp_path / "stt_dictionary.yaml"
    with open(dict_path, "w", encoding="utf-8") as f:
        yaml.dump(dict_content, f, allow_unicode=True, default_flow_style=False)
    return str(dict_path)


@pytest.fixture
def stt_dict(stt_dict_path):
    """Create a loaded STTDictionary instance."""
    return STTDictionary(stt_dict_path)


@pytest.fixture
def empty_dict_path(tmp_path):
    """Create an empty dictionary file."""
    dict_path = tmp_path / "empty_dict.yaml"
    dict_path.write_text("corrections: []\npatterns: []\n", encoding="utf-8")
    return str(dict_path)


# ---------------------------------------------------------------------------
# Loading tests
# ---------------------------------------------------------------------------
class TestLoading:
    """Test dictionary loading from YAML files."""

    def test_load_valid_dictionary(self, stt_dict):
        corrections = stt_dict.list_corrections()
        assert len(corrections) == 3
        assert corrections[0]["wrong"] == "みらい"
        assert corrections[0]["correct"] == "MIRAI"

    def test_load_patterns(self, stt_dict):
        patterns = stt_dict.list_patterns()
        assert len(patterns) == 2
        assert patterns[0]["pattern"] == "おでんわ"

    def test_load_nonexistent_file(self, tmp_path):
        dictionary = STTDictionary(str(tmp_path / "nonexistent.yaml"))
        assert dictionary.list_corrections() == []
        assert dictionary.list_patterns() == []

    def test_load_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")
        dictionary = STTDictionary(str(empty_file))
        assert dictionary.list_corrections() == []
        assert dictionary.list_patterns() == []

    def test_load_non_dict_yaml(self, tmp_path):
        list_file = tmp_path / "list.yaml"
        list_file.write_text("- item1\n- item2\n", encoding="utf-8")
        dictionary = STTDictionary(str(list_file))
        assert dictionary.list_corrections() == []
        assert dictionary.list_patterns() == []

    def test_load_invalid_yaml(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{{invalid: yaml: content", encoding="utf-8")
        dictionary = STTDictionary(str(bad_file))
        assert dictionary.list_corrections() == []
        assert dictionary.list_patterns() == []

    def test_load_empty_corrections(self, empty_dict_path):
        dictionary = STTDictionary(empty_dict_path)
        assert dictionary.list_corrections() == []
        assert dictionary.list_patterns() == []

    def test_load_null_corrections(self, tmp_path):
        """YAML with null corrections/patterns should not crash."""
        null_file = tmp_path / "null.yaml"
        null_file.write_text(
            "corrections: null\npatterns: null\n", encoding="utf-8"
        )
        dictionary = STTDictionary(str(null_file))
        assert dictionary.list_corrections() == []
        assert dictionary.list_patterns() == []


# ---------------------------------------------------------------------------
# Saving tests
# ---------------------------------------------------------------------------
class TestSaving:
    """Test dictionary saving to YAML files."""

    def test_save_and_reload(self, stt_dict, stt_dict_path):
        stt_dict.add_correction("てすと", "テスト", "テスト用")
        stt_dict.save()

        reloaded = STTDictionary(stt_dict_path)
        corrections = reloaded.list_corrections()
        wrong_values = [c["wrong"] for c in corrections]
        assert "てすと" in wrong_values

    def test_save_creates_parent_directories(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "dict.yaml"
        dictionary = STTDictionary(str(deep_path))
        dictionary.add_correction("foo", "bar")
        dictionary.save()

        assert deep_path.exists()
        reloaded = STTDictionary(str(deep_path))
        assert len(reloaded.list_corrections()) == 1

    def test_save_preserves_unicode(self, tmp_path):
        dict_path = tmp_path / "unicode.yaml"
        dictionary = STTDictionary(str(dict_path))
        dictionary.add_correction("こんにちわ", "こんにちは", "挨拶")
        dictionary.save()

        with open(dict_path, encoding="utf-8") as f:
            content = f.read()
        assert "こんにちわ" in content
        assert "こんにちは" in content


# ---------------------------------------------------------------------------
# Text correction tests
# ---------------------------------------------------------------------------
class TestCorrection:
    """Test text correction functionality."""

    def test_exact_correction(self, stt_dict):
        result = stt_dict.correct("みらいにでんわしました")
        assert "MIRAI" in result
        assert "みらい" not in result

    def test_multiple_corrections_in_text(self, stt_dict):
        result = stt_dict.correct("みらいのえーぴーあいを使う")
        assert "MIRAI" in result
        assert "API" in result

    def test_pattern_correction(self, stt_dict):
        result = stt_dict.correct("おでんわありがとうございます")
        assert "お電話" in result
        assert "おでんわ" not in result

    def test_pattern_with_backreference(self, stt_dict):
        result = stt_dict.correct("100えんです")
        assert "100円" in result

    def test_no_match(self, stt_dict):
        text = "何も修正されない文章です"
        assert stt_dict.correct(text) == text

    def test_empty_text(self, stt_dict):
        assert stt_dict.correct("") == ""

    def test_longest_match_first(self, stt_dict):
        """Longer corrections should be applied first to avoid partial matches."""
        result = stt_dict.correct("うぇるあいのみらいにでんわ")
        assert "WellAI" in result
        assert "MIRAI" in result

    def test_empty_dictionary_returns_original(self, empty_dict_path):
        dictionary = STTDictionary(empty_dict_path)
        text = "みらいにでんわしました"
        assert dictionary.correct(text) == text

    def test_correction_with_invalid_pattern(self, tmp_path):
        """Invalid regex patterns should be skipped without crashing."""
        dict_content = {
            "corrections": [],
            "patterns": [
                {"pattern": "[invalid", "replacement": "fixed"},
            ],
        }
        dict_path = tmp_path / "bad_pattern.yaml"
        with open(dict_path, "w", encoding="utf-8") as f:
            yaml.dump(dict_content, f, allow_unicode=True)

        dictionary = STTDictionary(str(dict_path))
        result = dictionary.correct("some text")
        assert result == "some text"


# ---------------------------------------------------------------------------
# CRUD operations - corrections
# ---------------------------------------------------------------------------
class TestCorrectionCRUD:
    """Test add/remove/list for corrections."""

    def test_add_correction(self, stt_dict):
        stt_dict.add_correction("ふぁいる", "ファイル", "IT用語")
        corrections = stt_dict.list_corrections()
        wrong_values = [c["wrong"] for c in corrections]
        assert "ふぁいる" in wrong_values

    def test_add_correction_updates_existing(self, stt_dict):
        stt_dict.add_correction("みらい", "Core", "英語表記")
        corrections = stt_dict.list_corrections()
        entry = next(c for c in corrections if c["wrong"] == "みらい")
        assert entry["correct"] == "Core"
        assert entry["note"] == "英語表記"
        # Should not add a duplicate
        assert sum(1 for c in corrections if c["wrong"] == "みらい") == 1

    def test_add_correction_without_note(self, stt_dict):
        stt_dict.add_correction("ぷろぐらむ", "プログラム")
        corrections = stt_dict.list_corrections()
        entry = next(c for c in corrections if c["wrong"] == "ぷろぐらむ")
        assert entry["note"] == ""

    def test_remove_correction(self, stt_dict):
        assert stt_dict.remove_correction("みらい") is True
        corrections = stt_dict.list_corrections()
        wrong_values = [c["wrong"] for c in corrections]
        assert "みらい" not in wrong_values

    def test_remove_nonexistent_correction(self, stt_dict):
        assert stt_dict.remove_correction("存在しない") is False

    def test_list_corrections_returns_copy(self, stt_dict):
        corrections = stt_dict.list_corrections()
        corrections.append({"wrong": "test", "correct": "test"})
        assert len(stt_dict.list_corrections()) != len(corrections)


# ---------------------------------------------------------------------------
# CRUD operations - patterns
# ---------------------------------------------------------------------------
class TestPatternCRUD:
    """Test add/remove/list for patterns."""

    def test_add_pattern(self, stt_dict):
        stt_dict.add_pattern(r"\bてすと\b", "テスト")
        patterns = stt_dict.list_patterns()
        pattern_values = [p["pattern"] for p in patterns]
        assert r"\bてすと\b" in pattern_values

    def test_add_pattern_updates_existing(self, stt_dict):
        stt_dict.add_pattern("おでんわ", "お電話番号")
        patterns = stt_dict.list_patterns()
        entry = next(p for p in patterns if p["pattern"] == "おでんわ")
        assert entry["replacement"] == "お電話番号"
        assert sum(1 for p in patterns if p["pattern"] == "おでんわ") == 1

    def test_add_invalid_pattern_raises(self, stt_dict):
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            stt_dict.add_pattern("[invalid", "replacement")

    def test_remove_pattern(self, stt_dict):
        assert stt_dict.remove_pattern("おでんわ") is True
        patterns = stt_dict.list_patterns()
        pattern_values = [p["pattern"] for p in patterns]
        assert "おでんわ" not in pattern_values

    def test_remove_nonexistent_pattern(self, stt_dict):
        assert stt_dict.remove_pattern("存在しない") is False

    def test_list_patterns_returns_copy(self, stt_dict):
        patterns = stt_dict.list_patterns()
        patterns.append({"pattern": "test", "replacement": "test"})
        assert len(stt_dict.list_patterns()) != len(patterns)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_correction_with_special_characters(self, tmp_path):
        dict_path = tmp_path / "special.yaml"
        dictionary = STTDictionary(str(dict_path))
        dictionary.add_correction("(かっこ)", "(括弧)")
        result = dictionary.correct("(かっこ)内のテキスト")
        assert "(括弧)" in result

    def test_overlapping_corrections(self, tmp_path):
        """When corrections overlap, longer ones should take priority."""
        dict_content = {
            "corrections": [
                {"wrong": "あい", "correct": "AI", "note": ""},
                {"wrong": "あいう", "correct": "AIU", "note": ""},
            ],
            "patterns": [],
        }
        dict_path = tmp_path / "overlap.yaml"
        with open(dict_path, "w", encoding="utf-8") as f:
            yaml.dump(dict_content, f, allow_unicode=True)

        dictionary = STTDictionary(str(dict_path))
        result = dictionary.correct("あいうえお")
        # Longer match "あいう" -> "AIU" should be applied first
        assert "AIU" in result

    def test_corrections_and_patterns_combined(self, stt_dict):
        result = stt_dict.correct("みらいにおでんわしました 500えんです")
        assert "MIRAI" in result
        assert "お電話" in result
        assert "500円" in result

    def test_correction_entry_with_missing_fields(self, tmp_path):
        """Entries with missing required fields should be skipped safely."""
        dict_content = {
            "corrections": [
                {"wrong": "てすと"},  # missing 'correct'
                {"correct": "テスト"},  # missing 'wrong'
                {"wrong": "", "correct": "nothing"},  # empty wrong
            ],
            "patterns": [
                {"replacement": "test"},  # missing 'pattern'
                {"pattern": ""},  # empty pattern
            ],
        }
        dict_path = tmp_path / "incomplete.yaml"
        with open(dict_path, "w", encoding="utf-8") as f:
            yaml.dump(dict_content, f, allow_unicode=True)

        dictionary = STTDictionary(str(dict_path))
        # Should not crash
        result = dictionary.correct("てすとのぶんしょう")
        assert isinstance(result, str)

    def test_reload_after_external_modification(self, stt_dict_path):
        dictionary = STTDictionary(stt_dict_path)
        assert len(dictionary.list_corrections()) == 3

        # Externally modify the file
        with open(stt_dict_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {"corrections": [{"wrong": "new", "correct": "NEW", "note": ""}], "patterns": []},
                f,
                allow_unicode=True,
            )

        dictionary.load()
        assert len(dictionary.list_corrections()) == 1
        assert dictionary.list_corrections()[0]["wrong"] == "new"


# ---------------------------------------------------------------------------
# Integration with VoskSTT (mocked)
# ---------------------------------------------------------------------------
class TestVoskSTTIntegration:
    """Test STTDictionary integration with VoskSTT."""

    @patch("src.stt.vosk_stt.SetLogLevel")
    @patch("src.stt.vosk_stt.KaldiRecognizer")
    @patch("src.stt.vosk_stt.Model")
    @patch("src.stt.vosk_stt.Path")
    def test_vosk_with_dictionary(
        self, mock_path_cls, mock_model, mock_recognizer_cls, mock_set_log, stt_dict
    ):
        from src.stt.vosk_stt import VoskSTT

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance

        stt = VoskSTT(
            model_path="models/vosk/vosk-model-small-ja-0.22",
            dictionary=stt_dict,
        )
        assert stt.dictionary is stt_dict

    @patch("src.stt.vosk_stt.SetLogLevel")
    @patch("src.stt.vosk_stt.KaldiRecognizer")
    @patch("src.stt.vosk_stt.Model")
    @patch("src.stt.vosk_stt.Path")
    def test_vosk_recognize_applies_correction(
        self, mock_path_cls, mock_model, mock_recognizer_cls, mock_set_log, stt_dict
    ):
        from src.stt.vosk_stt import VoskSTT

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance

        # Mock recognizer to return text with misrecognition
        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.FinalResult.return_value = json.dumps(
            {"text": "みらいにでんわしました"}
        )
        mock_recognizer_cls.return_value = mock_recognizer

        stt = VoskSTT(
            model_path="models/vosk/vosk-model-small-ja-0.22",
            dictionary=stt_dict,
        )

        audio = np.zeros(16000, dtype=np.int16)
        result = stt.recognize(audio)

        assert "MIRAI" in result
        assert "みらい" not in result

    @patch("src.stt.vosk_stt.SetLogLevel")
    @patch("src.stt.vosk_stt.KaldiRecognizer")
    @patch("src.stt.vosk_stt.Model")
    @patch("src.stt.vosk_stt.Path")
    def test_vosk_without_dictionary(
        self, mock_path_cls, mock_model, mock_recognizer_cls, mock_set_log
    ):
        from src.stt.vosk_stt import VoskSTT

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance

        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.FinalResult.return_value = json.dumps(
            {"text": "みらいにでんわしました"}
        )
        mock_recognizer_cls.return_value = mock_recognizer

        stt = VoskSTT(model_path="models/vosk/vosk-model-small-ja-0.22")

        audio = np.zeros(16000, dtype=np.int16)
        result = stt.recognize(audio)

        # Without dictionary, text should be unchanged
        assert result == "みらいにでんわしました"

    @patch("src.stt.vosk_stt.SetLogLevel")
    @patch("src.stt.vosk_stt.KaldiRecognizer")
    @patch("src.stt.vosk_stt.Model")
    @patch("src.stt.vosk_stt.Path")
    def test_vosk_recognize_empty_text_with_dictionary(
        self, mock_path_cls, mock_model, mock_recognizer_cls, mock_set_log, stt_dict
    ):
        from src.stt.vosk_stt import VoskSTT

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance

        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.FinalResult.return_value = json.dumps({"text": ""})
        mock_recognizer_cls.return_value = mock_recognizer

        stt = VoskSTT(
            model_path="models/vosk/vosk-model-small-ja-0.22",
            dictionary=stt_dict,
        )

        audio = np.zeros(16000, dtype=np.int16)
        result = stt.recognize(audio)

        assert result == ""


# ---------------------------------------------------------------------------
# Security tests (Codex review findings)
# ---------------------------------------------------------------------------
class TestRegexDoSPrevention:
    """Tests for regex DoS prevention in STT dictionary."""

    def test_long_pattern_rejected(self, tmp_path):
        from src.stt.dictionary import MAX_PATTERN_LENGTH

        dictionary = STTDictionary(str(tmp_path / "test.yaml"))
        long_pattern = "a" * (MAX_PATTERN_LENGTH + 1)

        with pytest.raises(ValueError, match="Pattern too long"):
            dictionary.add_pattern(long_pattern, "b")

    def test_max_length_pattern_accepted(self, tmp_path):
        from src.stt.dictionary import MAX_PATTERN_LENGTH

        dictionary = STTDictionary(str(tmp_path / "test.yaml"))
        ok_pattern = "a" * MAX_PATTERN_LENGTH

        dictionary.add_pattern(ok_pattern, "b")
        assert len(dictionary.list_patterns()) == 1


class TestReplaceAll:
    """Tests for bulk replace_all method used by UI save."""

    def test_replace_all_overwrites_corrections(self, tmp_path):
        dictionary = STTDictionary(str(tmp_path / "test.yaml"))
        dictionary.add_correction("old", "OLD")
        dictionary.add_correction("other", "OTHER")

        dictionary.replace_all(
            corrections=[{"wrong": "new", "correct": "NEW", "note": "replaced"}],
            patterns=[],
        )

        result = dictionary.list_corrections()
        assert len(result) == 1
        assert result[0]["wrong"] == "new"
        assert result[0]["correct"] == "NEW"

    def test_replace_all_overwrites_patterns(self, tmp_path):
        dictionary = STTDictionary(str(tmp_path / "test.yaml"))
        dictionary.add_pattern("old_pat", "OLD")

        dictionary.replace_all(
            corrections=[],
            patterns=[{"pattern": "new_pat", "replacement": "NEW"}],
        )

        result = dictionary.list_patterns()
        assert len(result) == 1
        assert result[0]["pattern"] == "new_pat"

    def test_replace_all_with_empty_clears(self, tmp_path):
        dictionary = STTDictionary(str(tmp_path / "test.yaml"))
        dictionary.add_correction("a", "A")
        dictionary.add_pattern("b", "B")

        dictionary.replace_all(corrections=[], patterns=[])

        assert dictionary.list_corrections() == []
        assert dictionary.list_patterns() == []

    def test_replace_all_then_save_and_reload(self, tmp_path):
        dict_path = tmp_path / "test.yaml"
        dictionary = STTDictionary(str(dict_path))

        dictionary.replace_all(
            corrections=[{"wrong": "x", "correct": "X", "note": ""}],
            patterns=[{"pattern": "y", "replacement": "Y"}],
        )
        dictionary.save()

        reloaded = STTDictionary(str(dict_path))
        assert len(reloaded.list_corrections()) == 1
        assert reloaded.list_corrections()[0]["wrong"] == "x"
        assert len(reloaded.list_patterns()) == 1
        assert reloaded.list_patterns()[0]["pattern"] == "y"

    def test_replace_all_applies_corrections(self, tmp_path):
        dictionary = STTDictionary(str(tmp_path / "test.yaml"))
        dictionary.replace_all(
            corrections=[{"wrong": "abc", "correct": "ABC", "note": ""}],
            patterns=[],
        )

        assert dictionary.correct("abc def") == "ABC def"


class TestThreadSafeDictionary:
    """Tests for thread safety in STT dictionary."""

    def test_has_lock(self, tmp_path):
        import threading

        dictionary = STTDictionary(str(tmp_path / "test.yaml"))
        assert hasattr(dictionary, "_lock")
        assert isinstance(dictionary._lock, type(threading.Lock()))
