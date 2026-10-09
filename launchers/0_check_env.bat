@echo off
REM shows whether PyTorch sees the GPU
cd /d "%~dp0.."
python scripts\check_env.py
set "check_status=%errorlevel%"
pause
exit /b %check_status%
