"""The agentx-trace-eval launcher's staleness fixes: before these, ensure_installed downloaded
once and then ran that binary forever - a July install silently served a July engine months
later, with no version stamp, no way to ask for an update, and the dashboard fetched from
releases/latest independently of the engine (so the two halves of one install could skew)."""

import io
import sys

import pytest

from agentx import cli


@pytest.fixture
def install_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "INSTALL_DIR", tmp_path)
    return tmp_path


def _fake_install(calls):
    def fake(version="latest"):
        calls.append(version)

    return fake


def test_tag_resolved_from_redirected_release_url():
    url = "https://github.com/AgentX-ai/AgentX-trace-eval/releases/download/v0.3.0/agentx_darwin_arm64.tar.gz"
    assert cli._tag_from_release_url(url) == "v0.3.0"
    assert cli._tag_from_release_url("https://example.com/not-a-release") is None


def test_existing_install_is_not_redownloaded(install_dir, monkeypatch):
    (install_dir / "agentx-server").write_text("bin")
    calls = []
    monkeypatch.setattr(cli, "_install", _fake_install(calls))
    cli.ensure_installed(version="latest")
    assert calls == []


def test_force_reinstalls_over_existing(install_dir, monkeypatch):
    (install_dir / "agentx-server").write_text("bin")
    calls = []
    monkeypatch.setattr(cli, "_install", _fake_install(calls))
    cli.ensure_installed(version="latest", force=True)
    assert calls == ["latest"]


def test_changed_version_pin_reinstalls(install_dir, monkeypatch):
    (install_dir / "agentx-server").write_text("bin")
    (install_dir / ".version").write_text("v0.2.0\n")
    calls = []
    monkeypatch.setattr(cli, "_install", _fake_install(calls))

    # Same pin: no download. Different pin: download. "latest": trusts what's there.
    cli.ensure_installed(version="v0.2.0")
    assert calls == []
    cli.ensure_installed(version="v0.3.0")
    assert calls == ["v0.3.0"]
    calls.clear()
    cli.ensure_installed(version="latest")
    assert calls == []


def test_update_notice_names_both_versions(install_dir, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_latest_release_tag", lambda: "v0.9.0")
    cli._maybe_print_update_notice("v0.2.0")
    err = capsys.readouterr().err
    assert "v0.2.0" in err and "v0.9.0" in err and "--update" in err


def test_update_notice_handles_unstamped_install(install_dir, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_latest_release_tag", lambda: None)
    cli._maybe_print_update_notice(None)
    assert "--update" in capsys.readouterr().err


def test_update_notice_is_silent_when_current(install_dir, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_latest_release_tag", lambda: "v0.2.0")
    cli._maybe_print_update_notice("v0.2.0")
    assert capsys.readouterr().err == ""


def test_latest_release_tag_fails_open(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(cli.requests, "get", boom)
    assert cli._latest_release_tag() is None
