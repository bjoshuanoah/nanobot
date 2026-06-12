"""Tests for nanobot.agent.tools.audit — AuditTool and AuditEvent."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.tools.audit import (
    AuditEvent,
    AuditTool,
    AuditToolConfig,
    _matches_scope,
    load_audit_config,
)

# ---------------------------------------------------------------------------
# AuditEvent
# ---------------------------------------------------------------------------


class TestAuditEvent:
    def test_defaults(self):
        ev = AuditEvent(tool="read_file", status="ok", detail="done")
        assert ev.iteration == -1
        assert ev.session_key is None

    def test_full_fields(self):
        ev = AuditEvent(
            tool="read_file",
            status="error",
            detail="not found",
            iteration=3,
            session_key="abc123",
        )
        assert ev.tool == "read_file"
        assert ev.status == "error"
        assert ev.iteration == 3
        assert ev.session_key == "abc123"


# ---------------------------------------------------------------------------
# Scope matching
# ---------------------------------------------------------------------------


class TestMatchesScope:
    @pytest.mark.parametrize(
        "tool, patterns, expected",
        [
            ("read_file", ["*"], True),
            ("read_file", ["read_*"], True),
            ("read_file", ["exec_*"], False),
            ("web_fetch", ["web_*", "exec_*"], True),
            ("exec", ["*"], True),
            ("exec", [], False),
        ],
    )
    def test_glob_matching(self, tool, patterns, expected):
        assert _matches_scope(tool, patterns) is expected


# ---------------------------------------------------------------------------
# AuditToolConfig & load_audit_config
# ---------------------------------------------------------------------------


class TestAuditToolConfig:
    def test_defaults(self):
        cfg = AuditToolConfig()
        assert cfg.enable is False
        assert cfg.scope == ["*"]
        assert cfg.transport == "log"

    def test_custom(self):
        cfg = AuditToolConfig(enable=True, scope=["exec_*"], transport="callback")
        assert cfg.enable is True
        assert cfg.scope == ["exec_*"]
        assert cfg.transport == "callback"


class TestLoadAuditConfig:
    def test_missing_section(self):
        """Config without audit section → defaults."""

        class Cfg:
            pass

        cfg = load_audit_config(Cfg())
        assert cfg.enable is False

    def test_present_section(self):
        """Config with audit section → values propagated."""

        class AuditSection:
            enable = True
            scope = ["web_*"]
            transport = "bus"

        class Cfg:
            audit = AuditSection()

        cfg = load_audit_config(Cfg())
        assert cfg.enable is True
        assert cfg.scope == ["web_*"]
        assert cfg.transport == "bus"


# ---------------------------------------------------------------------------
# AuditTool.record()
# ---------------------------------------------------------------------------


class TestAuditToolRecord:
    @pytest.mark.asyncio
    async def test_disabled_is_noop(self, caplog):
        """When enable=False, record() does nothing — no log, no callback."""
        tool = AuditTool(config=AuditToolConfig(enable=False))
        ev = AuditEvent(tool="exec", status="ok", detail="ran")
        with caplog.at_level(logging.INFO, logger="nanobot.agent.tools.audit"):
            await tool.record(ev)
        assert not caplog.records

    @pytest.mark.asyncio
    async def test_scope_filter(self, caplog):
        """Tool not in scope → silently skipped."""
        tool = AuditTool(config=AuditToolConfig(enable=True, scope=["web_*"]))
        ev = AuditEvent(tool="exec", status="ok", detail="ran")
        with caplog.at_level(logging.INFO, logger="nanobot.agent.tools.audit"):
            await tool.record(ev)
        assert not caplog.records

    @pytest.mark.asyncio
    async def test_log_transport_emits_log(self, caplog):
        """Default log transport → logger.info with audit event details."""
        tool = AuditTool(config=AuditToolConfig(enable=True, transport="log"))
        ev = AuditEvent(
            tool="read_file",
            status="ok",
            detail="success",
            iteration=5,
        )
        with caplog.at_level(logging.INFO, logger="nanobot.agent.tools.audit"):
            await tool.record(ev)
        assert len(caplog.records) == 1
        assert "read_file" in caplog.records[0].message
        assert "ok" in caplog.records[0].message

    @pytest.mark.asyncio
    async def test_callback_transport(self):
        """Callback transport → awaitable called with the event."""
        callback = AsyncMock()
        tool = AuditTool(
            config=AuditToolConfig(enable=True, transport="callback"),
            _callback=callback,
        )
        ev = AuditEvent(tool="write_file", status="error", detail="denied")
        await tool.record(ev)
        callback.assert_awaited_once_with(ev)

    @pytest.mark.asyncio
    async def test_callback_transport_without_callback(self, caplog):
        """Callback transport with no callback set → event dropped silently."""
        tool = AuditTool(
            config=AuditToolConfig(enable=True, transport="callback"),
            _callback=None,
        )
        ev = AuditEvent(tool="write_file", status="ok", detail="wrote")
        with caplog.at_level(logging.DEBUG, logger="nanobot.agent.tools.audit"):
            await tool.record(ev)
        # No crash, no log records at INFO level for callback transport
        assert not any(r.levelno >= logging.INFO for r in caplog.records)

    @pytest.mark.asyncio
    async def test_bus_transport_logs_debug(self, caplog):
        """Bus transport logs at debug level (no real bus wired in unit tests)."""
        tool = AuditTool(config=AuditToolConfig(enable=True, transport="bus"))
        ev = AuditEvent(tool="exec", status="ok", detail="ran")
        with caplog.at_level(logging.DEBUG, logger="nanobot.agent.tools.audit"):
            await tool.record(ev)
        assert any("audit bus" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_detail_truncation_in_log(self, caplog):
        """Log transport truncates detail to 120 chars in the log line."""
        tool = AuditTool(config=AuditToolConfig(enable=True, transport="log"))
        long_detail = "x" * 200
        ev = AuditEvent(tool="exec", status="ok", detail=long_detail)
        with caplog.at_level(logging.INFO, logger="nanobot.agent.tools.audit"):
            await tool.record(ev)
        logged = caplog.records[0].message
        # The format string uses %.120s, so detail is truncated to 120 chars
        assert long_detail[:120] in logged


# ---------------------------------------------------------------------------
# AuditTool.from_config()
# ---------------------------------------------------------------------------


class TestAuditToolFromConfig:
    def test_factory_defaults(self):
        """Config without audit section → AuditTool with defaults."""

        class Cfg:
            pass

        tool = AuditTool.from_config(Cfg())
        assert tool.config.enable is False
        assert tool.config.scope == ["*"]
        assert tool.config.transport == "log"

    def test_factory_with_callback(self):
        """Callback is propagated through factory."""

        async def cb(ev: AuditEvent) -> None:
            pass

        class AuditSection:
            enable = True
            scope = ["exec_*"]
            transport = "callback"

        class Cfg:
            audit = AuditSection()

        tool = AuditTool.from_config(Cfg(), callback=cb)
        assert tool.config.enable is True
        assert tool._callback is cb
