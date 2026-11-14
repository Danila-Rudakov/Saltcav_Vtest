#!/usr/bin/env python3
"""Запуск веб-интерфейса и сервера saltcav в одну команду.

Использование:
    python run_web.py [порт, по умолчанию 8080]
"""

import sys
from pathlib import Path

# Добавляем папку src в sys.path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from saltcav.web.server import launch_server

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    launch_server(host="127.0.0.1", port=port)

