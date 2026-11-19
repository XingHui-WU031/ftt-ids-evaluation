@echo off
REM Step 2: smoke test, then the full Plan-A pipeline on UNSW-NB15. Safe to re-run: finished runs are skipped.
cd /d "%~dp0.."
if not exist logs mkdir logs
set PY=python
set LOG=logs\step2.log
echo ===== start %date% %time% ===== >> %LOG%
%PY% run.py smoke >> %LOG% 2>&1
if errorlevel 1 (
  echo ===== SMOKE TEST FAILED - stopped ===== >> %LOG%
  echo Smoke test failed. Tell Claude.
  pause
  exit /b 1
)
%PY% run.py tune      --dataset unsw >> %LOG% 2>&1
%PY% run.py train     --dataset unsw --group main >> %LOG% 2>&1
%PY% run.py train     --dataset unsw --group ablation >> %LOG% 2>&1
%PY% run.py train     --dataset unsw_noshortcut --group shortcut >> %LOG% 2>&1
%PY% run.py aggregate --dataset unsw >> %LOG% 2>&1
%PY% run.py aggregate --dataset unsw_noshortcut >> %LOG% 2>&1
%PY% run.py explain   --dataset unsw --model proposed >> %LOG% 2>&1
%PY% run.py explain   --dataset unsw --model thesis_ftt >> %LOG% 2>&1
%PY% run.py profile   --dataset unsw >> %LOG% 2>&1
echo ===== STEP2 FINISHED %date% %time% ===== >> %LOG%
echo Done. You can close this window.
pause
