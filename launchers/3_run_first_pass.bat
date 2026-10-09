@echo off
REM E1 with seed 0 only (ResNet1d, CNN-BiLSTM, ECG-ViT on 12 leads). Write down how long each run takes.
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0.."
python scripts\run_experiments.py --exp e1 --seeds 0 || goto :err
python scripts\summarize.py || goto :err
echo FIRST PASS COMPLETE
pause
exit /b 0

:err
echo FIRST PASS FAILED with exit code %errorlevel%
pause
exit /b 1
