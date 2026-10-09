@echo off
REM all tables and figures, the conclusion table, ensembles/subgroups, explainability figures.
cd /d "%~dp0.."
python scripts\summarize.py --results results --out results\summary
python scripts\conclusion_table.py --results results --out results\summary
python scripts\ensemble_subgroup.py --results results --out results\summary
for %%m in (resnet1d_12lead resnet1d_leadI bilstm_12lead vit_12lead laresnet_12lead laresnet_leadI) do python scripts\explain.py --run results\%%m_frac1_seed0
echo.
pause
