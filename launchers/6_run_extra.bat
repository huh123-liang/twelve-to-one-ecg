@echo off
REM E1b ablations and E5 label efficiency (about 18 runs). Run after 5_run_core.bat.
cd /d "%~dp0.."
python scripts\run_experiments.py --exp e1b
python scripts\run_experiments.py --exp e5
pause
