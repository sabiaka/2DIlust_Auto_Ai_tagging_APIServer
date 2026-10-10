@echo off
setlocal
set "SERVER_HOST=0.0.0.0"
call "%~dp0Start-Server.bat"
exit /b %errorlevel%
