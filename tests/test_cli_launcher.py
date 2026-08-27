"""The agentx-trace-eval launcher's version pinning: each SDK release names the engine release
it was tested against (agentx.version.ENGINE_VERSION) and the launcher converges the local
install to it, so `pip install -U agentx-python` upgrades the engine to the matching pair.
Before this, the launcher downloaded "latest" once and then ran that binary forever - a July
install silently served a July engine months later, with no stamp and no way to update."""

import pytest

from agentx import cli
from agentx.version import ENGINE_VERSION


@pytest.fixture
def install_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "INSTALL_DIR", tmp_path)
    return tmp_path


def _fake_install(calls):
    def fake(version="latest"):
        calls.append(version)

    return fake


def test_engine_pin_is_a_release_tag():
    # The pin the SDK ships must look like a real trace-eval release tag (v<series>.<n>), not
    # "latest" - "latest" as a pin would silently disable the whole tested-pair contract.
    assert cli.ENGINE_VERSION == ENGINE_VERSION
    import re

    assert re.fullmatch(r"v\d+\.\d+\.\d+", ENGINE_VERSION)


def test_resolve_version_prefers_env_override(monkeypatch):
    monkeypatch.delenv("AGENTX_TRACE_EVAL_VERSION", raising=False)
    assert cli.resolve_version() == ENGINE_VERSION
    monkeypatch.setenv("AGENTX_TRACE_EVAL_VERSION", "latest")
    assert cli.resolve_version() == "latest"
    monkeypatch.setenv("AGENTX_TRACE_EVAL_VERSION", "v9.9.9")
    assert cli.resolve_version() == "v9.9.9"


def test_tag_resolved_from_redirected_release_url():
    url = "https://github.com/AgentX-ai/AgentX-trace-eval/releases/download/v0.3.1/agentx_darwin_arm64.tar.gz"
    assert cli._tag_from_release_url(url) == "v0.3.1"
    assert cli._tag_from_release_url("https://example.com/not-a-release") is None


def test_install_matching_pin_is_not_redownloaded(install_dir, monkeypatch):
    (install_dir / "agentx-server").write_text("bin")
    (install_dir / ".version").write_text(ENGINE_VERSION + "\n")
    calls = []
    monkeypatch.setattr(cli, "_install", _fake_install(calls))
    cli.ensure_installed(version=ENGINE_VERSION)
    assert calls == []


def test_stamp_differing_from_pin_converges(install_dir, monkeypatch):
    # The SDK-upgrade path: binaries exist but were installed for an older pin (or pre-stamp,
    # where .version is missing entirely) - the launcher re-installs the pinned pair.
    (install_dir / "agentx-server").write_text("bin")
    (install_dir / ".version").write_text("v0.2.0\n")
    calls = []
    monkeypatch.setattr(cli, "_install", _fake_install(calls))
    cli.ensure_installed(version=ENGINE_VERSION)
    assert calls == [ENGINE_VERSION]

    (install_dir / ".version").unlink()
    calls.clear()
    cli.ensure_installed(version=ENGINE_VERSION)
    assert calls == [ENGINE_VERSION]


def test_latest_mode_trusts_existing_install(install_dir, monkeypatch):
    (install_dir / "agentx-server").write_text("bin")
    (install_dir / ".version").write_text("v0.2.0\n")
    calls = []
    monkeypatch.setattr(cli, "_install", _fake_install(calls))
    cli.ensure_installed(version="latest")
    assert calls == []


def test_force_reinstalls_over_existing(install_dir, monkeypatch):
    (install_dir / "agentx-server").write_text("bin")
    (install_dir / ".version").write_text(ENGINE_VERSION + "\n")
    calls = []
    monkeypatch.setattr(cli, "_install", _fake_install(calls))
    cli.ensure_installed(version=ENGINE_VERSION, force=True)
    assert calls == [ENGINE_VERSION]


def test_update_notice_names_both_versions(install_dir, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_latest_release_tag", lambda: "v0.9.0")
    cli._maybe_print_update_notice("v0.3.1")
    err = capsys.readouterr().err
    assert "v0.3.1" in err and "v0.9.0" in err and "--update" in err


def test_update_notice_handles_unstamped_install(install_dir, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_latest_release_tag", lambda: None)
    cli._maybe_print_update_notice(None)
    assert "--update" in capsys.readouterr().err


def test_update_notice_is_silent_when_current(install_dir, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_latest_release_tag", lambda: "v0.3.1")
    cli._maybe_print_update_notice("v0.3.1")
    assert capsys.readouterr().err == ""


def test_latest_release_tag_fails_open(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(cli.requests, "get", boom)
    assert cli._latest_release_tag() is None
