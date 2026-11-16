@echo off
REM Round 3 (2026-11-15): reviewer-requested experiments. About 7-9 hours on the RTX 5060. Safe to re-run: finished runs are skipped.
REM   A  no-dedup control on UNSW-NB15
REM   B  host-disjoint evaluation on TON_IoT (5 folds x 3 seeds)
REM   C  second attribution method (Integrated Gradients) + expected gradients with 2000 samples
REM   D  retraining-based faithfulness (ROAR): remove SHAP top-k vs random-k features and retrain
REM   E  larger hyperparameter search for XGBoost and FTT-IDS
cd /d "%~dp0"
if not exist logs mkdir logs
set PY=..\Python_all\.venv\Scripts\python.exe
set R=results_r3
set LOG=logs\step5.log
echo ===== start %date% %time% ===== >> %LOG%
%PY% run.py check --results %R% >> %LOG% 2>&1
echo ----- A no-dedup ----- >> %LOG%
%PY% run.py train --dataset unsw_nodedup --models xgboost rf proposed ft_transformer --results %R% >> %LOG% 2>&1
echo ===== STAGE A FINISHED %date% %time% ===== >> %LOG%
echo ----- B host-disjoint TON_IoT ----- >> %LOG%
for %%k in (0 1 2 3 4) do %PY% run.py train --dataset toniot_host%%k --models xgboost rf mlp ft_transformer proposed --seeds 42 43 44 --results %R% >> %LOG% 2>&1
echo ===== STAGE B FINISHED %date% %time% ===== >> %LOG%
echo ----- C explanation checks ----- >> %LOG%
%PY% run.py explain_r3 --dataset unsw --model proposed --results %R% >> %LOG% 2>&1
echo ===== STAGE C FINISHED %date% %time% ===== >> %LOG%
echo ----- D ROAR ----- >> %LOG%
for %%d in (top5 top10 rnd5a rnd5b rnd10a rnd10b) do %PY% run.py train --dataset unsw_roar_%%d --models xgboost proposed --seeds 42 43 44 --results %R% >> %LOG% 2>&1
echo ===== STAGE D FINISHED %date% %time% ===== >> %LOG%
echo ----- E search ----- >> %LOG%
%PY% run.py search --dataset unsw --models xgboost proposed --trials 20 --results %R% >> %LOG% 2>&1
echo ===== STEP5 FINISHED %date% %time% ===== >> %LOG%
echo Done. You can close this window.
pause
