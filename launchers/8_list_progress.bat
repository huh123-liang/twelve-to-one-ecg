@echo off
REM Shows which of the 102 planned runs are done and which are still to do.
cd /d "%~dp0.."
python scripts\run_experiments.py --exp all --list
pause
