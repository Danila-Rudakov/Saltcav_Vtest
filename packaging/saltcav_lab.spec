# -*- mode: python ; coding: utf-8 -*-
# Лёгкая сборка лабораторного веб-интерфейса SALTCAV LAB (PyInstaller):
#   pyinstaller packaging/saltcav_lab.spec --noconfirm --clean
# Результат по умолчанию — один файл dist/saltcav_lab(.exe). Папкой (быстрее запуск, меньше
# ложных срабатываний антивируса): задать переменную окружения SALTCAV_ONEDIR=1.
# В отличие от packaging/saltcav.spec (полное приложение: Qt, Gmsh, MKL) здесь только то, что
# нужно модели сечения и веб-интерфейсу: numpy, scipy, pydantic, pyyaml.
# PyInstaller собирает под ту систему, на которой запущен: .exe — только на Windows.

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

ROOT = Path(SPECPATH).parent  # noqa: F821 — задаёт PyInstaller
sys.path.insert(0, str(ROOT / "src"))

ONEDIR = os.environ.get("SALTCAV_ONEDIR") == "1"

# страницы интерфейса (static/) лежат внутри пакета saltcav.web; метаданные — для __version__
datas = collect_data_files("saltcav.web")
datas += copy_metadata("saltcav")
for rel in ("configs/lab", "docs/lab"):  # примеры опытов и описание модели — рядом с программой
    if (ROOT / rel).is_dir():
        datas.append((str(ROOT / rel), rel))

hidden = [
    "saltcav.web.server",
    "saltcav.core.lab_config",
    "saltcav.leaching.section",
    "saltcav.leaching.axisym",
    "saltcav.leaching.material",
    "saltcav.leaching.brine_column",
    "scipy.interpolate",
    "yaml",
]
EXCLUDES = ["PySide6", "PyQt5", "PyQt6", "matplotlib", "pyvista", "vtk", "gmsh", "numba",
            "sklearn", "tkinter", "IPython", "jupyter", "notebook", "pytest", "sphinx", "mypy"]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "saltcav_lab_web.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    hiddenimports=hidden,
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

if ONEDIR:
    exe = EXE(  # noqa: F821
        pyz, a.scripts, [], exclude_binaries=True, name="saltcav_lab",
        console=True, upx=False,
    )
    coll = COLLECT(exe, a.binaries, a.datas, name="saltcav_lab", upx=False)  # noqa: F821
else:
    exe = EXE(  # noqa: F821
        pyz, a.scripts, a.binaries, a.datas, [], name="saltcav_lab",
        console=True, upx=False,
    )
