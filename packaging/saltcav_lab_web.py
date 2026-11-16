"""Точка входа лабораторного веб-интерфейса SALTCAV LAB в собранном приложении (PyInstaller).

Запускает локальный сервер на свободном порту и открывает интерфейс в браузере. Окно консоли
остаётся открытым, пока работает программа: закрыть окно — остановить сервер.
"""

from __future__ import annotations

import multiprocessing
import socket
import sys
import threading
import webbrowser

PREFERRED_PORT = 8080


def _free_port(preferred: int) -> int:
    """Порт 8080, если свободен, иначе любой свободный."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return int(s.getsockname()[1])
    raise RuntimeError("не найден свободный порт")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # кириллица в консоли Windows
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else _free_port(PREFERRED_PORT)
    from saltcav.web.server import launch_server

    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, webbrowser.open, args=(f"http://127.0.0.1:{port}",)).start()
    launch_server(port=port)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
