@echo off
rem Build saltcav_lab.exe (SALTCAV LAB web interface) with PyInstaller. Run on Windows.
rem Usage: double-click, or from Git Bash:  cmd.exe //c packaging\\build_lab_exe.bat
rem Set SALTCAV_ONEDIR=1 before running to get a folder instead of a single file.
setlocal
cd /d "%~dp0.."
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version || (echo Python 3.11+ not found. Install it from python.org and tick "Add to PATH". & goto :fail)
%PY% -m venv .venv-lab-exe || goto :fail
call .venv-lab-exe\Scripts\activate.bat || goto :fail
python -m pip install --upgrade pip || goto :fail
python -m pip install "numpy>=1.26" "scipy>=1.11" "pydantic>=2.6" "pyyaml>=6.0" "threadpoolctl>=3.1" "pyinstaller>=6.10" || goto :fail
python -m pip install --no-deps -e . || goto :fail
pyinstaller packaging\saltcav_lab.spec --noconfirm --clean || goto :fail
echo.
if "%SALTCAV_ONEDIR%"=="1" (echo DONE: dist\saltcav_lab\saltcav_lab.exe) else (echo DONE: dist\saltcav_lab.exe)
pause
exit /b 0
:fail
echo BUILD FAILED - see messages above.
pause
exit /b 1
