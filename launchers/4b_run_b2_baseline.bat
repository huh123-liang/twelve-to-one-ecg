@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
REM B2 shift-invariant feature baseline (CPU, a few minutes). Does not touch B0/B1 results.
cd /d "%~dp0.."
python scripts\baselines.py --models b2
pause
