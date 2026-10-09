@echo off
REM prior and logistic-regression baselines (CPU, a few minutes)
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0.."
python scripts\baselines.py || goto :err
echo BASELINES COMPLETE
pause
exit /b 0

:err
echo BASELINES FAILED with exit code %errorlevel%
pause
exit /b 1
