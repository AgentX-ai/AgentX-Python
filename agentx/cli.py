"""
`agentx-trace-eval` — thin launcher for AgentX's self-hostable governance engine (Trace,
Evaluate, Monitor), published separately at github.com/AgentX-ai/AgentX-trace-eval (a Go CLI
wrapping a Bun-compiled TypeScript engine, not Python). That compiled engine binary is tens of
megabytes; most `pip install agentx-python` installs are just this SDK talking to the hosted
AgentX SaaS and would never touch it, so it isn't bundled in this package. Instead, this command
downloads the matching release into ~/.agentx/bin the first time it's needed (mirroring
AgentX-trace-eval's own install.sh) and then hands off to the real `agentx-server` binary.

Usage:
    agentx-trace-eval --dev
    agentx-trace-eval --port 5000 --db-url postgres://...
"""

import os
import platform
import shutil
import stat
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Tuple

import requests

REPO = "AgentX-ai/AgentX-trace-eval"
INSTALL_DIR = Path(os.environ.get("AGENTX_INSTALL_DIR", str(Path.home() / ".agentx" / "bin")))
_BIN_NAMES = ("agentx", "agentx-server", "agentx-engine")


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


def _download(url: str, dest: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 16):
            f.write(chunk)


def _extract_tar(archive: Path, dest_dir: Path) -> None:
    with tarfile.open(archive) as tar:
        try:
            # filter="data" (PEP 706, Python 3.12+) rejects absolute paths/symlink escapes.
            # Belt-and-suspenders here since these are trusted release assets built by our own CI
            # (see REPO above), not arbitrary user-supplied archives.
            tar.extractall(dest_dir, filter="data")
        except TypeError:
            tar.extractall(dest_dir)  # Python < 3.12: filter kwarg doesn't exist yet


def _install(version: str = "latest") -> None:
    os_name, arch = _platform_tag()
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"agentx-trace-eval: downloading agentx ({os_name}/{arch})...", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        archive = tmp_dir / "agentx.tar.gz"
        url = _release_url(f"agentx_{os_name}_{arch}.tar.gz", version)
        try:
            _download(url, archive)
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

    if os.environ.get("AGENTX_TRACE_EVAL_SKIP_WEB"):
        return

    # Best-effort: the dashboard is a separate, platform-independent asset (see
    # AgentX-trace-eval's README's "Dashboard release"). Missing it shouldn't block getting the
    # engine running headless, so a failure here warns and continues rather than raising.
    print("agentx-trace-eval: downloading dashboard...", file=sys.stderr)
    web_dir = INSTALL_DIR / "web"
    with tempfile.TemporaryDirectory() as tmp:
        web_archive = Path(tmp) / "agentx-web.tar.gz"
        web_url = _release_url("agentx-web.tar.gz", version)
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


def ensure_installed(version: str = "latest") -> Path:
    """Downloads agentx-server (+ its engine) into ~/.agentx/bin if not already present there.
    Returns the path to the agentx-server executable. Set AGENTX_INSTALL_DIR to change where
    this looks/installs; set AGENTX_TRACE_EVAL_VERSION to pin a release tag instead of latest."""
    server_path = INSTALL_DIR / "agentx-server"
    if not server_path.exists():
        _install(version=version)
    return server_path


def main() -> None:
    version = os.environ.get("AGENTX_TRACE_EVAL_VERSION", "latest")
    server_path = ensure_installed(version=version)
    if not server_path.exists():
        raise SystemExit(f"agentx-trace-eval: {server_path} still missing after install, giving up")

    # os.execv replaces this process rather than spawning a subprocess: signals, stdio, and the
    # exit code all pass straight through to agentx-server, same as invoking it directly.
    os.execv(str(server_path), [str(server_path)] + sys.argv[1:])


if __name__ == "__main__":
    main()
