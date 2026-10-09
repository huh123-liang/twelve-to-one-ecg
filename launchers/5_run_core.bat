@echo off
REM E1 + E2 + E3 + E4 with seeds 0-2 (the "must" experiments, about 84 runs).
REM Safe to stop and restart: finished runs are skipped. Keep the laptop plugged in and disable sleep.
cd /d "%~dp0.."
python scripts\run_experiments.py --exp core
python scripts\summarize.py
pause
