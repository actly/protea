"""Tests for ring1.crystallizer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ring1.crystallizer import Crystallizer, CrystallizationResult


def _make_config(**overrides):
    cfg = MagicMock()
    cfg.claude_api_key = overrides.get("api_key", "sk-test")
    cfg.claude_model = overrides.get("model", "test-model")
    cfg.claude_max_tokens = overrides.get("max_tokens", 4096)
    return cfg


def _make_skill_store(skills=None):
    store = MagicMock()
    store.get_active.return_value = skills or []
    store.get_by_name.return_value = None
    store.count_active.return_value = len(skills) if skills else 0
    store.get_least_used.return_value = []
    store.add.return_value = 1
    store.update.return_value = True
    return store


SOURCE = "import os\ndef main(): pass  # PROTEA_HEARTBEAT"


class TestCrystallizeCreate:
    """Crystallizer should create new skills correctly."""

    def test_create_new_skill(self):
        config = _make_config()
        store = _make_skill_store()
        llm_resp = (
            '{"action": "create", "name": "web_dashboard", '
            '"description": "Web dashboard", '
            '"prompt_template": "HTTP server pattern", '
            '"tags": ["web"]}'
        )

        config.get_llm_client.return_value.send_message.return_value = llm_resp
        c = Crystallizer(config, store)
        result = c.crystallize(SOURCE, "output", generation=5)

        assert result.action == "create"
        assert result.skill_name == "web_dashboard"
        store.add.assert_called_once()
        call_kwargs = store.add.call_args
        assert call_kwargs[1]["source"] == "crystallized"
        assert call_kwargs[1]["source_code"] == SOURCE

    def test_create_evicts_when_full(self):
        skills = [{"name": f"s{i}", "description": "d", "tags": []} for i in range(5)]
        config = _make_config()
        store = _make_skill_store(skills)
        store.count_active.return_value = 5
        store.get_least_used.return_value = [{"name": "s0"}]

        llm_resp = (
            '{"action": "create", "name": "new_skill", '
            '"description": "New", "prompt_template": "t", "tags": []}'
        )

        config.get_llm_client.return_value.send_message.return_value = llm_resp
        c = Crystallizer(config, store)
        result = c.crystallize(SOURCE, "", generation=1, skill_cap=5)

        assert result.action == "create"
        store.deactivate.assert_called_once_with("s0")
        store.add.assert_called_once()

    def test_create_duplicate_name_converts_to_update(self):
        config = _make_config()
        store = _make_skill_store()
        store.get_by_name.return_value = {"name": "existing", "description": "old"}

        llm_resp = (
            '{"action": "create", "name": "existing", '
            '"description": "New desc", "prompt_template": "t", "tags": []}'
        )

        config.get_llm_client.return_value.send_message.return_value = llm_resp
        c = Crystallizer(config, store)
        result = c.crystallize(SOURCE, "", generation=1)

        assert result.action == "update"
        assert result.skill_name == "existing"
        store.update.assert_called_once()
        store.add.assert_not_called()


class TestCrystallizeUpdate:
    """Crystallizer should update existing skills correctly."""

    def test_update_existing_skill(self):
        config = _make_config()
        store = _make_skill_store()
        store.get_by_name.return_value = {"name": "web_dashboard", "description": "old"}

        llm_resp = (
            '{"action": "update", "existing_name": "web_dashboard", '
            '"description": "Updated", "prompt_template": "t", "tags": ["web"]}'
        )

        config.get_llm_client.return_value.send_message.return_value = llm_resp
        c = Crystallizer(config, store)
        result = c.crystallize(SOURCE, "", generation=2)

        assert result.action == "update"
        assert result.skill_name == "web_dashboard"
        store.update.assert_called_once()
        call_kwargs = store.update.call_args
        assert call_kwargs[1]["source_code"] == SOURCE

    def test_update_nonexistent_becomes_skip(self):
        config = _make_config()
        store = _make_skill_store()
        store.get_by_name.return_value = None

        llm_resp = (
            '{"action": "update", "existing_name": "nonexistent", '
            '"description": "d", "prompt_template": "t", "tags": []}'
        )

        config.get_llm_client.return_value.send_message.return_value = llm_resp
        c = Crystallizer(config, store)
        result = c.crystallize(SOURCE, "", generation=1)

        assert result.action == "skip"
        assert "not found" in result.reason


class TestCrystallizeSkip:
    """Crystallizer should handle skip responses."""

    def test_skip_returns_reason(self):
        config = _make_config()
        store = _make_skill_store()

        llm_resp = '{"action": "skip", "reason": "already covered by web_dashboard"}'

        config.get_llm_client.return_value.send_message.return_value = llm_resp
        c = Crystallizer(config, store)
        result = c.crystallize(SOURCE, "", generation=1)

        assert result.action == "skip"
        assert "already covered" in result.reason


class TestCrystallizeErrors:
    """Crystallizer should handle errors gracefully."""

    def test_llm_error(self):
        config = _make_config()
        store = _make_skill_store()

        from ring1.llm_base import LLMError

        config.get_llm_client.return_value.send_message.side_effect = LLMError("timeout")
        c = Crystallizer(config, store)
        result = c.crystallize(SOURCE, "", generation=1)

        assert result.action == "error"
        assert "LLM error" in result.reason

    def test_parse_failure(self):
        config = _make_config()
        store = _make_skill_store()

        config.get_llm_client.return_value.send_message.return_value = "not valid json"
        c = Crystallizer(config, store)
        result = c.crystallize(SOURCE, "", generation=1)

        assert result.action == "error"
        assert "parse" in result.reason

    def test_missing_name_in_create(self):
        config = _make_config()
        store = _make_skill_store()

        llm_resp = '{"action": "create", "description": "d", "prompt_template": "t"}'

        config.get_llm_client.return_value.send_message.return_value = llm_resp
        c = Crystallizer(config, store)
        result = c.crystallize(SOURCE, "", generation=1)

        assert result.action == "error"
        assert "missing" in result.reason

    def test_store_read_error(self):
        config = _make_config()
        store = _make_skill_store()
        store.get_active.side_effect = Exception("db locked")

        c = Crystallizer(config, store)
        result = c.crystallize(SOURCE, "", generation=1)

        assert result.action == "error"
        assert "skill store" in result.reason


class TestCrystallizationResult:
    """CrystallizationResult namedtuple."""

    def test_fields(self):
        r = CrystallizationResult(action="create", skill_name="test", reason="OK")
        assert r.action == "create"
        assert r.skill_name == "test"
        assert r.reason == "OK"
