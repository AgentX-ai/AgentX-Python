"""
`agentx-trace-eval` - thin launcher for AgentX's self-hostable governance engine (Trace,
Evaluate, Monitor), published separately at github.com/AgentX-ai/AgentX-trace-eval (a Go CLI
wrapping a Bun-compiled TypeScript engine, not Python). That compiled engine binary is tens of
megabytes; most `pip install agentx-python` installs are just this SDK talking to the hosted
AgentX SaaS and would never touch it, so it isn't bundled in this package. Instead, this command
downloads the release into ~/.agentx/bin the first time it's needed (mirroring
AgentX-trace-eval's own install.sh) and then hands off to the real `agentx-server` binary.

Versioning: each SDK release pins the engine release it was tested against
(agentx.version.ENGINE_VERSION); the launcher installs that pin and converges to it - if the
stamped install (~/.agentx/bin/.version) differs from the pin, it re-installs, so upgrading the
SDK upgrades the engine to the matching pair. `--update` force-reinstalls the resolved version
(recovering a broken or pre-stamp install). Set AGENTX_TRACE_EVAL_VERSION to a tag or `latest`
to override the pin; in `latest` mode the launcher trusts whatever is installed, prints a
fail-open notice when GitHub has something newer, and `--update` fetches it.

Usage:
    agentx-trace-eval --dev
    agentx-trace-eval --update --dev
    AGENTX_TRACE_EVAL_VERSION=latest agentx-trace-eval --update --dev
    agentx-trace-eval --port 5000 --db-url postgres://...
"""

import os
import platform
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import requests

from agentx.version import ENGINE_VERSION

REPO = "AgentX-ai/AgentX-trace-eval"
INSTALL_DIR = Path(os.environ.get("AGENTX_INSTALL_DIR", str(Path.home() / ".agentx" / "bin")))
_BIN_NAMES = ("agentx", "agentx-server", "agentx-engine")
_VERSION_STAMP = ".version"


def _platform_tag() -> Tuple[str, str]:
    system = platform.system()
    if system == "Darwin":
        os_name = "darwin"
    elif system == "Linux":
        os_name = "linux"
    else:
        raise SystemExit(
            f"agentx-trace-eval: unsupported OS {system!r} (self-host currently supports macOS and Linux)"
        )

    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        arch = "arm64"
    elif machine in ("x86_64", "amd64"):
        arch = "amd64"
    else:
        raise SystemExit(f"agentx-trace-eval: unsupported architecture {machine!r}")

    return os_name, arch


def _release_url(asset: str, version: str) -> str:
    if version == "latest":
        return f"https://github.com/{REPO}/releases/latest/download/{asset}"
    return f"https://github.com/{REPO}/releases/download/{version}/{asset}"


def _tag_from_release_url(url: str) -> Optional[str]:
    # GitHub redirects releases/latest/download/... to releases/download/<tag>/...; the final
    # URL is the one place the resolved tag shows up without a second API call.
    match = re.search(r"/releases/download/([^/]+)/", url)
    return match.group(1) if match else None


def _download(url: str, dest: Path) -> Optional[str]:
    """Downloads url to dest; returns the release tag resolved from the final (post-redirect)
    URL, or None if it can't be determined."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return _tag_from_release_url(response.url)


def _extract_tar(archive: Path, dest_dir: Path) -> None:
    with tarfile.open(archive) as tar:
        try:
            # filter="data" (PEP 706, Python 3.12+) rejects absolute paths/symlink escapes.
            # Belt-and-suspenders here since these are trusted release assets built by our own CI
            # (see REPO above), not arbitrary user-supplied archives.
            tar.extractall(dest_dir, filter="data")
        except TypeError:
            tar.extractall(dest_dir)  # Python < 3.12: filter kwarg doesn't exist yet


def _stamped_version() -> Optional[str]:
    try:
        return (INSTALL_DIR / _VERSION_STAMP).read_text().strip() or None
    except OSError:
        return None


def _latest_release_tag() -> Optional[str]:
    """The newest published tag, or None when it can't be determined (offline, rate-limited).
    Fail-open by design: an update NOTICE must never break launching the engine."""
    try:
        response = requests.get(f"https://api.github.com/repos/{REPO}/releases/latest", timeout=3)
        response.raise_for_status()
        tag = response.json().get("tag_name")
        return tag if isinstance(tag, str) and tag else None
    except Exception:
        return None


def resolve_version() -> str:
    """The engine version this launcher should be running: the AGENTX_TRACE_EVAL_VERSION
    override (a tag, or `latest`) when set, else the SDK's tested pin."""
    return os.environ.get("AGENTX_TRACE_EVAL_VERSION") or ENGINE_VERSION


def _install(version: str = "latest") -> None:
    os_name, arch = _platform_tag()
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"agentx-trace-eval: downloading agentx {version} ({os_name}/{arch})...", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        archive = tmp_dir / "agentx.tar.gz"
        url = _release_url(f"agentx_{os_name}_{arch}.tar.gz", version)
        try:
            resolved_tag = _download(url, archive)
        except requests.HTTPError as exc:
            raise SystemExit(
                f"agentx-trace-eval: failed to download {url} ({exc}).\n"
                "No published release found; see AgentX-trace-eval's README for building from source."
            )

        _extract_tar(archive, tmp_dir)

        found_any = False
        for name in _BIN_NAMES:
            src = tmp_dir / name
            if not src.exists():
                continue
            found_any = True
            dest = INSTALL_DIR / name
            shutil.move(str(src), str(dest))
            dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        if not found_any:
            raise SystemExit(f"agentx-trace-eval: downloaded archive from {url} didn't contain any of {_BIN_NAMES}")

    stamp = resolved_tag or (version if version != "latest" else None)
    if stamp:
        (INSTALL_DIR / _VERSION_STAMP).write_text(stamp + "\n")
        print(f"agentx-trace-eval: installed {stamp}", file=sys.stderr)

    if os.environ.get("AGENTX_TRACE_EVAL_SKIP_WEB"):
        return

    # Best-effort: the dashboard is a separate, platform-independent asset (see
    # AgentX-trace-eval's README's "Dashboard release"). Missing it shouldn't block getting the
    # engine running headless, so a failure here warns and continues rather than raising.
    print("agentx-trace-eval: downloading dashboard...", file=sys.stderr)
    web_dir = INSTALL_DIR / "web"
    with tempfile.TemporaryDirectory() as tmp:
        web_archive = Path(tmp) / "agentx-web.tar.gz"
        # Pin the dashboard to the tag the engine actually resolved to, so the two halves of one
        # install can't skew (releases/latest assets are updated independently).
        web_url = _release_url("agentx-web.tar.gz", stamp or version)
        try:
            _download(web_url, web_archive)
        except requests.HTTPError:
            print(
                f"agentx-trace-eval: no dashboard bundle found at {web_url}, continuing without one",
                file=sys.stderr,
            )
            return
        if web_dir.exists():
            shutil.rmtree(web_dir)
        web_dir.mkdir(parents=True)
        _extract_tar(web_archive, web_dir)


def ensure_installed(version: Optional[str] = None, force: bool = False) -> Path:
    """Installs agentx-server (+ its engine and dashboard) into ~/.agentx/bin when it's missing
    there, when `force` is set, or when `version` is a specific tag that differs from the
    stamped install - the convergence that makes an SDK upgrade carry its tested engine along.
    `version` defaults to resolve_version() (the AGENTX_TRACE_EVAL_VERSION override, else the
    SDK's pin); pass "latest" to trust whatever is installed and only fetch when missing.
    Returns the path to the agentx-server executable. Set AGENTX_INSTALL_DIR to change where
    this looks/installs."""
    if version is None:
        version = resolve_version()
    server_path = INSTALL_DIR / "agentx-server"
    stamped = _stamped_version()
    pin_mismatch = version != "latest" and stamped != version
    if force or pin_mismatch or not server_path.exists():
        _install(version=version)
    return server_path


def _maybe_print_update_notice(installed: Optional[str]) -> None:
    """Only relevant in `latest` mode, where nothing converges automatically - a quick fail-open
    check tells the user when the world has moved on. (Pinned mode needs no notice: a pin
    mismatch re-installs instead of nagging.)"""
    latest = _latest_release_tag()
    if installed is None:
        print(
            "agentx-trace-eval: installed engine version unknown"
            + (f" (newest release is {latest})" if latest else "")
            + "; run with --update to fetch the newest release",
            file=sys.stderr,
        )
    elif latest and latest != installed:
        print(
            f"agentx-trace-eval: engine {installed} is installed but {latest} is available; "
            "run with --update to upgrade",
            file=sys.stderr,
        )


def main() -> None:
    args = sys.argv[1:]
    force_update = "--update" in args
    # --update belongs to this launcher, not to agentx-server - consume it before the handoff.
    args = [a for a in args if a != "--update"]

    version = resolve_version()
    server_path = ensure_installed(version=version, force=force_update)
    if not server_path.exists():
        raise SystemExit(f"agentx-trace-eval: {server_path} still missing after install, giving up")

    if not force_update and version == "latest":
        _maybe_print_update_notice(_stamped_version())

    # os.execv replaces this process rather than spawning a subprocess: signals, stdio, and the
    # exit code all pass straight through to agentx-server, same as invoking it directly.
    os.execv(str(server_path), [str(server_path)] + args)


if __name__ == "__main__":
    main()
