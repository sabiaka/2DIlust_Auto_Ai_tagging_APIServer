@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. Follow the Python setup in README.md first.
  pause
  exit /b 1
)
echo Starting the Python API. Press Ctrl+C to stop.
".venv\Scripts\python.exe" -u run_server.py
set "serverExit=%errorlevel%"
if not "%serverExit%"=="0" pause
exit /b %serverExit%
