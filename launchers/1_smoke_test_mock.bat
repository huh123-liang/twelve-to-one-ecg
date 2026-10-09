@echo off
REM the WHOLE v2 pipeline on a small FAKE dataset (about 10-20 min).
REM Results in results_mock\ are meaningless - this only checks that every part of the code runs.
cd /d "%~dp0.."
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
python scripts\make_mock_ptbxl.py --out data\mock-ptb-xl --n 600 || goto :err
python scripts\prepare_data.py --ptbxl data\mock-ptb-xl --cache data\mock-cache || goto :err
python scripts\baselines.py --cache data\mock-cache --out results_mock || goto :err
python scripts\run_experiments.py --exp all --seeds 0 --extra="--epochs 1 --batch 32 --cache data/mock-cache --out results_mock" || goto :err
python scripts\summarize.py --results results_mock --out results_mock\summary || goto :err
python scripts\conclusion_table.py --results results_mock --out results_mock\summary --n_boot 100 || goto :err
python scripts\ensemble_subgroup.py --results results_mock --out results_mock\summary || goto :err
python scripts\explain.py --run results_mock\vit_12lead_frac1_seed0 || goto :err
python scripts\explain.py --run results_mock\laresnet_leadI_frac1_seed0 || goto :err
echo.
echo SMOKE TEST PASSED
pause
exit /b 0
:err
echo SMOKE TEST FAILED
pause
exit /b 1
