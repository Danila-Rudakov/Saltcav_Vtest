"""Лаунчер собранного лабораторного приложения (packaging/saltcav_lab_web.py)."""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load() -> object:
    spec = importlib.util.spec_from_file_location(
        "saltcav_lab_web", ROOT / "packaging" / "saltcav_lab_web.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_free_port_prefers_8080_and_falls_back() -> None:
    mod = _load()
    port = mod._free_port(8080)  # type: ignore[attr-defined]
    assert 1024 <= port <= 65535
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        taken = busy.getsockname()[1]
        busy.listen(1)
        other = mod._free_port(taken)  # type: ignore[attr-defined]
        assert other != taken


def test_build_files_present() -> None:
    for name in ("saltcav_lab.spec", "saltcav_lab_web.py", "build_lab_exe.bat"):
        assert (ROOT / "packaging" / name).is_file()
