"""
Tests for the KnowledgeManager module and its integration with OllamaClient.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm.knowledge import KnowledgeManager, _derive_title, _sanitize_filename


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def knowledge_dir(tmp_path):
    """Create a temporary knowledge directory with sample files."""
    kdir = tmp_path / "knowledge"
    kdir.mkdir()

    (kdir / "company_info.md").write_text(
        "# 会社情報\n会社名: テスト株式会社\n営業時間: 9:00-18:00",
        encoding="utf-8",
    )
    (kdir / "faq.md").write_text(
        "# FAQ\n## Q: 営業時間は？\nA: 9時から18時です。",
        encoding="utf-8",
    )

    return str(kdir)


@pytest.fixture
def empty_knowledge_dir(tmp_path):
    """Create an empty temporary knowledge directory."""
    kdir = tmp_path / "empty_knowledge"
    kdir.mkdir()
    return str(kdir)


@pytest.fixture
def manager(knowledge_dir):
    """Create a KnowledgeManager with sample data."""
    return KnowledgeManager(knowledge_dir=knowledge_dir)


# ---------------------------------------------------------------------------
# _sanitize_filename tests
# ---------------------------------------------------------------------------
class TestSanitizeFilename:
    """Tests for the _sanitize_filename utility function."""

    def test_basic_ascii(self):
        assert _sanitize_filename("hello_world") == "hello_world"

    def test_spaces_replaced(self):
        assert _sanitize_filename("hello world") == "hello_world"

    def test_special_chars_replaced(self):
        result = _sanitize_filename("file@name#test!")
        assert "@" not in result
        assert "#" not in result
        assert "!" not in result

    def test_consecutive_underscores_collapsed(self):
        assert _sanitize_filename("a   b") == "a_b"

    def test_leading_trailing_underscores_stripped(self):
        assert _sanitize_filename("  hello  ") == "hello"

    def test_empty_string_returns_untitled(self):
        assert _sanitize_filename("") == "untitled"

    def test_all_special_chars_returns_untitled(self):
        assert _sanitize_filename("@#$%") == "untitled"

    def test_japanese_characters_preserved(self):
        result = _sanitize_filename("会社情報")
        assert "会社情報" in result

    def test_unicode_normalization(self):
        # Full-width A -> half-width A via NFKC
        result = _sanitize_filename("\uff21\uff22\uff23")
        assert result == "ABC"


# ---------------------------------------------------------------------------
# _derive_title tests
# ---------------------------------------------------------------------------
class TestDeriveTitle:
    """Tests for the _derive_title utility function."""

    def test_strips_md_extension(self):
        assert _derive_title("company_info.md") == "company_info"

    def test_no_extension(self):
        assert _derive_title("readme") == "readme"

    def test_multiple_dots(self):
        assert _derive_title("file.v2.md") == "file.v2"


# ---------------------------------------------------------------------------
# KnowledgeManager initialization tests
# ---------------------------------------------------------------------------
class TestKnowledgeManagerInit:
    """Tests for KnowledgeManager initialization and directory handling."""

    def test_creates_missing_directory(self, tmp_path):
        new_dir = tmp_path / "nonexistent" / "nested" / "knowledge"
        manager = KnowledgeManager(knowledge_dir=str(new_dir))
        assert new_dir.exists()
        assert len(manager._entries) == 0

    def test_loads_existing_files(self, manager):
        assert len(manager._entries) == 2
        assert "company_info.md" in manager._entries
        assert "faq.md" in manager._entries

    def test_empty_directory(self, empty_knowledge_dir):
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        assert len(manager._entries) == 0


# ---------------------------------------------------------------------------
# load_all tests
# ---------------------------------------------------------------------------
class TestLoadAll:
    """Tests for the load_all method."""

    def test_reload_picks_up_new_files(self, manager, knowledge_dir):
        kdir = Path(knowledge_dir)
        (kdir / "new_entry.md").write_text("New content", encoding="utf-8")

        manager.load_all()

        assert "new_entry.md" in manager._entries
        assert len(manager._entries) == 3

    def test_reload_removes_deleted_files(self, manager, knowledge_dir):
        kdir = Path(knowledge_dir)
        (kdir / "faq.md").unlink()

        manager.load_all()

        assert "faq.md" not in manager._entries
        assert len(manager._entries) == 1

    def test_ignores_non_md_files(self, knowledge_dir):
        kdir = Path(knowledge_dir)
        (kdir / "notes.txt").write_text("Not markdown", encoding="utf-8")
        (kdir / "data.json").write_text("{}", encoding="utf-8")

        manager = KnowledgeManager(knowledge_dir=knowledge_dir)

        assert "notes.txt" not in manager._entries
        assert "data.json" not in manager._entries
        assert len(manager._entries) == 2

    def test_handles_unreadable_file(self, knowledge_dir):
        """Files that fail to read should be skipped without crashing."""
        kdir = Path(knowledge_dir)
        bad_file = kdir / "bad.md"
        bad_file.write_bytes(b"\x80\x81\x82\x83")

        # Should not raise, just log an error and skip
        manager = KnowledgeManager(knowledge_dir=knowledge_dir)
        # The two valid files should still be loaded
        assert len(manager._entries) >= 2


# ---------------------------------------------------------------------------
# get_context tests
# ---------------------------------------------------------------------------
class TestGetContext:
    """Tests for the get_context method."""

    def test_returns_empty_string_when_no_entries(self, empty_knowledge_dir):
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        assert manager.get_context() == ""

    def test_contains_all_entry_titles(self, manager):
        context = manager.get_context()
        assert "=== company_info ===" in context
        assert "=== faq ===" in context

    def test_contains_entry_content(self, manager):
        context = manager.get_context()
        assert "テスト株式会社" in context
        assert "9時から18時" in context

    def test_sections_separated_by_double_newline(self, manager):
        context = manager.get_context()
        assert "\n\n=== " in context

    def test_max_chars_truncation(self, manager):
        context = manager.get_context(max_chars=50)
        assert len(context) <= 50

    def test_truncation_at_section_boundary(self, knowledge_dir):
        """When truncating, prefer to cut at a section boundary."""
        kdir = Path(knowledge_dir)
        # Add a third entry to make truncation more interesting
        (kdir / "extra.md").write_text("x" * 100, encoding="utf-8")

        manager = KnowledgeManager(knowledge_dir=knowledge_dir)
        full_context = manager.get_context(max_chars=99999)

        # Use a max_chars that cuts into the third section
        truncated = manager.get_context(max_chars=len(full_context) - 10)

        # Should not end in the middle of a section header
        assert truncated.count("===") % 2 == 0  # Headers come in pairs (=== title ===)

    def test_large_max_chars_returns_full_content(self, manager):
        context_default = manager.get_context()
        context_large = manager.get_context(max_chars=999999)
        assert context_default == context_large


# ---------------------------------------------------------------------------
# add_entry tests
# ---------------------------------------------------------------------------
class TestAddEntry:
    """Tests for the add_entry method."""

    def test_creates_file(self, empty_knowledge_dir):
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        filename = manager.add_entry("test_entry", "Some content")

        assert filename == "test_entry.md"
        filepath = Path(empty_knowledge_dir) / filename
        assert filepath.exists()
        assert filepath.read_text(encoding="utf-8") == "Some content"

    def test_adds_to_entries_dict(self, empty_knowledge_dir):
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        filename = manager.add_entry("my_entry", "Content here")

        assert filename in manager._entries
        assert manager._entries[filename] == "Content here"

    def test_sanitizes_title(self, empty_knowledge_dir):
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        filename = manager.add_entry("My Entry Title!", "Content")

        assert " " not in filename
        assert "!" not in filename
        assert filename.endswith(".md")

    def test_empty_title_raises(self, empty_knowledge_dir):
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        with pytest.raises(ValueError, match="Title cannot be empty"):
            manager.add_entry("", "Content")

    def test_whitespace_only_title_raises(self, empty_knowledge_dir):
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        with pytest.raises(ValueError, match="Title cannot be empty"):
            manager.add_entry("   ", "Content")

    def test_empty_content_raises(self, empty_knowledge_dir):
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        with pytest.raises(ValueError, match="Content cannot be empty"):
            manager.add_entry("title", "")

    def test_whitespace_only_content_raises(self, empty_knowledge_dir):
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        with pytest.raises(ValueError, match="Content cannot be empty"):
            manager.add_entry("title", "   ")

    def test_japanese_title(self, empty_knowledge_dir):
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        filename = manager.add_entry("会社情報", "内容")

        assert filename == "会社情報.md"
        assert manager.get_entry(filename) == "内容"


# ---------------------------------------------------------------------------
# update_entry tests
# ---------------------------------------------------------------------------
class TestUpdateEntry:
    """Tests for the update_entry method."""

    def test_updates_existing_file(self, manager, knowledge_dir):
        result = manager.update_entry("company_info.md", "Updated content")

        assert result is True
        assert manager._entries["company_info.md"] == "Updated content"

        filepath = Path(knowledge_dir) / "company_info.md"
        assert filepath.read_text(encoding="utf-8") == "Updated content"

    def test_returns_false_for_nonexistent(self, manager):
        result = manager.update_entry("nonexistent.md", "Content")
        assert result is False

    def test_empty_content_raises(self, manager):
        with pytest.raises(ValueError, match="Content cannot be empty"):
            manager.update_entry("company_info.md", "")


# ---------------------------------------------------------------------------
# remove_entry tests
# ---------------------------------------------------------------------------
class TestRemoveEntry:
    """Tests for the remove_entry method."""

    def test_removes_existing_file(self, manager, knowledge_dir):
        result = manager.remove_entry("faq.md")

        assert result is True
        assert "faq.md" not in manager._entries

        filepath = Path(knowledge_dir) / "faq.md"
        assert not filepath.exists()

    def test_returns_false_for_nonexistent(self, manager):
        result = manager.remove_entry("nonexistent.md")
        assert result is False

    def test_remove_and_get_context(self, manager):
        manager.remove_entry("faq.md")
        context = manager.get_context()
        assert "faq" not in context
        assert "company_info" in context


# ---------------------------------------------------------------------------
# list_entries tests
# ---------------------------------------------------------------------------
class TestListEntries:
    """Tests for the list_entries method."""

    def test_returns_all_entries(self, manager):
        entries = manager.list_entries()
        assert len(entries) == 2

    def test_entry_structure(self, manager):
        entries = manager.list_entries()
        for entry in entries:
            assert "filename" in entry
            assert "title" in entry
            assert "size" in entry
            assert "preview" in entry

    def test_title_derived_from_filename(self, manager):
        entries = manager.list_entries()
        filenames_and_titles = {e["filename"]: e["title"] for e in entries}
        assert filenames_and_titles["company_info.md"] == "company_info"
        assert filenames_and_titles["faq.md"] == "faq"

    def test_preview_limited_to_100_chars(self, empty_knowledge_dir):
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        long_content = "x" * 200
        manager.add_entry("long", long_content)

        entries = manager.list_entries()
        assert len(entries[0]["preview"]) <= 100

    def test_preview_replaces_newlines(self, manager):
        entries = manager.list_entries()
        for entry in entries:
            assert "\n" not in entry["preview"]

    def test_empty_directory_returns_empty_list(self, empty_knowledge_dir):
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        assert manager.list_entries() == []


# ---------------------------------------------------------------------------
# get_entry tests
# ---------------------------------------------------------------------------
class TestGetEntry:
    """Tests for the get_entry method."""

    def test_returns_content(self, manager):
        content = manager.get_entry("company_info.md")
        assert content is not None
        assert "テスト株式会社" in content

    def test_returns_none_for_nonexistent(self, manager):
        assert manager.get_entry("nonexistent.md") is None


# ---------------------------------------------------------------------------
# OllamaClient integration tests
# ---------------------------------------------------------------------------
class TestOllamaClientKnowledgeIntegration:
    """Tests for OllamaClient integration with KnowledgeManager."""

    def _make_client(self, knowledge_manager=None):
        """Create an OllamaClient with a mocked ollama.Client."""
        with patch("src.llm.ollama_client.Client"):
            from src.llm.ollama_client import OllamaClient

            return OllamaClient(
                model="test-model",
                knowledge_manager=knowledge_manager,
            )

    def test_no_knowledge_manager_no_context_injection(self):
        client = self._make_client(knowledge_manager=None)
        messages = client._build_messages("質問です", None, False)

        user_msg = messages[-1]["content"]
        assert user_msg == "質問です"
        assert "参考情報" not in user_msg

    def test_knowledge_manager_injects_context(self, manager):
        client = self._make_client(knowledge_manager=manager)
        messages = client._build_messages("営業時間は？", None, False)

        user_msg = messages[-1]["content"]
        assert "参考情報" in user_msg
        assert "テスト株式会社" in user_msg
        assert "営業時間は？" in user_msg

    def test_explicit_context_overrides_knowledge(self, manager):
        client = self._make_client(knowledge_manager=manager)
        messages = client._build_messages(
            "質問です", "明示的なコンテキスト", False
        )

        user_msg = messages[-1]["content"]
        assert "明示的なコンテキスト" in user_msg
        assert "テスト株式会社" not in user_msg

    def test_empty_knowledge_no_context_injection(self, empty_knowledge_dir):
        empty_manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        client = self._make_client(knowledge_manager=empty_manager)
        messages = client._build_messages("質問です", None, False)

        user_msg = messages[-1]["content"]
        assert user_msg == "質問です"
        assert "参考情報" not in user_msg

    def test_system_message_always_first(self, manager):
        client = self._make_client(knowledge_manager=manager)
        messages = client._build_messages("質問", None, False)

        assert messages[0]["role"] == "system"

    def test_history_included_with_knowledge(self, manager):
        client = self._make_client(knowledge_manager=manager)
        client._conversation_history = [
            {"role": "user", "content": "前の質問"},
            {"role": "assistant", "content": "前の回答"},
        ]

        messages = client._build_messages("新しい質問", None, True)

        assert len(messages) == 4  # system + 2 history + user
        assert messages[1]["content"] == "前の質問"
        assert messages[2]["content"] == "前の回答"
        assert "参考情報" in messages[3]["content"]


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_large_knowledge_file(self, empty_knowledge_dir):
        """Large files should be loaded and truncated in context."""
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        large_content = "A" * 5000
        manager.add_entry("large", large_content)

        context = manager.get_context(max_chars=2000)
        assert len(context) <= 2000

    def test_max_chars_zero(self, manager):
        """max_chars=0 should return empty or very short string."""
        context = manager.get_context(max_chars=0)
        assert len(context) == 0

    def test_concurrent_add_and_list(self, empty_knowledge_dir):
        """Add entries and immediately list them."""
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)

        manager.add_entry("entry1", "Content 1")
        manager.add_entry("entry2", "Content 2")
        manager.add_entry("entry3", "Content 3")

        entries = manager.list_entries()
        assert len(entries) == 3

    def test_add_then_remove_then_list(self, empty_knowledge_dir):
        """Full lifecycle: add, verify, remove, verify."""
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)

        filename = manager.add_entry("temp", "Temporary content")
        assert len(manager.list_entries()) == 1

        manager.remove_entry(filename)
        assert len(manager.list_entries()) == 0

    def test_update_reflects_in_context(self, manager):
        """Updated content should appear in get_context."""
        manager.update_entry("company_info.md", "更新された会社情報")
        context = manager.get_context()
        assert "更新された会社情報" in context
        assert "テスト株式会社" not in context

    def test_special_characters_in_content(self, empty_knowledge_dir):
        """Content with special markdown characters should be handled."""
        manager = KnowledgeManager(knowledge_dir=empty_knowledge_dir)
        content = "# Header\n- **bold**\n- *italic*\n- `code`\n- [link](url)"
        filename = manager.add_entry("special", content)

        retrieved = manager.get_entry(filename)
        assert retrieved == content

    def test_gitkeep_ignored(self, tmp_path):
        """The .gitkeep file in knowledge dir should be ignored."""
        kdir = tmp_path / "knowledge"
        kdir.mkdir()
        (kdir / ".gitkeep").write_text("", encoding="utf-8")
        (kdir / "info.md").write_text("Info", encoding="utf-8")

        manager = KnowledgeManager(knowledge_dir=str(kdir))
        assert len(manager._entries) == 1
        assert "info.md" in manager._entries


# ---------------------------------------------------------------------------
# Security tests (Codex review findings)
# ---------------------------------------------------------------------------
class TestPathTraversal:
    """Tests for path traversal prevention in knowledge manager."""

    def test_update_rejects_path_traversal(self, manager):
        """Filenames with '../' should be rejected."""
        result = manager.update_entry("../config/settings.yaml", "malicious")
        assert result is False

    def test_remove_rejects_path_traversal(self, manager):
        """Filenames with '../' should be rejected."""
        result = manager.remove_entry("../config/settings.yaml")
        assert result is False

    def test_get_entry_rejects_path_traversal(self, manager):
        """Filenames with '../' should return None."""
        result = manager.get_entry("../config/settings.yaml")
        assert result is None

    def test_update_rejects_slash_in_filename(self, manager):
        result = manager.update_entry("foo/bar.md", "content")
        assert result is False

    def test_remove_rejects_slash_in_filename(self, manager):
        result = manager.remove_entry("foo/bar.md")
        assert result is False

    def test_update_rejects_dotfile(self, manager):
        result = manager.update_entry(".hidden", "content")
        assert result is False

    def test_get_entry_rejects_empty_filename(self, manager):
        result = manager.get_entry("")
        assert result is None


class TestThreadSafeKnowledge:
    """Tests for thread safety in knowledge manager."""

    def test_has_lock(self, manager):
        import threading
        assert hasattr(manager, "_lock")
        assert isinstance(manager._lock, type(threading.Lock()))
