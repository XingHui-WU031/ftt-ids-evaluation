@echo off
setlocal
cd /d "%~dp0.."
REM Run in the activated Python environment used for the original experiments.
REM Optional first argument: folder containing Train_Test_Network.csv.
set "TON_DATA=%~1"
if "%TON_DATA%"=="" set "TON_DATA=..\数据集\TON_IoT"
python run.py host_strict --data-dir "%TON_DATA%" --results results_strict_host --models rf xgboost mlp ft_transformer proposed --seeds 42 43 44
if errorlevel 1 exit /b 1
python analysis/08_strict_host_summary.py --results results_strict_host
if errorlevel 1 exit /b 1
REM Sensitivity run: retain all within-host duplicates as well.
python run.py host_strict --data-dir "%TON_DATA%" --dedup-policy none --results results_strict_host_nodedup --models rf xgboost mlp ft_transformer proposed --seeds 42 43 44
if errorlevel 1 exit /b 1
python analysis/08_strict_host_summary.py --results results_strict_host_nodedup --suffix _nodedup
exit /b %errorlevel%
