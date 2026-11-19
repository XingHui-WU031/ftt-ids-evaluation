@echo off
REM Step 3: remove all TTL features (E10b) + test-set sensitivity analysis. About 1 hour.
cd /d "%~dp0.."
set PY=python
set LOG=logs\step3.log
echo ===== start %date% %time% ===== >> %LOG%
%PY% run.py sensitivity --dataset unsw >> %LOG% 2>&1
%PY% run.py train       --dataset unsw_nottl --group shortcut >> %LOG% 2>&1
%PY% run.py aggregate   --dataset unsw_nottl >> %LOG% 2>&1
%PY% run.py sensitivity --dataset unsw_nottl >> %LOG% 2>&1
%PY% run.py sensitivity --dataset unsw_noshortcut >> %LOG% 2>&1
echo ===== STEP3 FINISHED %date% %time% ===== >> %LOG%
echo Done. You can close this window.
pause
