@echo off
REM needs the real PTB-XL in data\ptb-xl (ptbxl_database.csv directly inside).
cd /d "%~dp0.."
python scripts\prepare_data.py --ptbxl data\ptb-xl --cache data\cache || goto :err
python scripts\eda.py --cache data\cache --out results\eda || goto :err
pause
exit /b 0
:err
pause
exit /b 1
