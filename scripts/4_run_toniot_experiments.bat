@echo off
REM Step 4: TON_IoT (second dataset). Run AFTER step 3 has finished. Safe to re-run.
cd /d "%~dp0.."
set PY=python
set LOG=logs\step4.log
echo ===== start %date% %time% ===== >> %LOG%
%PY% run.py tune      --dataset toniot >> %LOG% 2>&1
%PY% run.py train     --dataset toniot --group main >> %LOG% 2>&1
%PY% run.py train     --dataset toniot --group ablation >> %LOG% 2>&1
%PY% run.py aggregate --dataset toniot >> %LOG% 2>&1
%PY% run.py explain   --dataset toniot --model proposed >> %LOG% 2>&1
%PY% run.py profile   --dataset toniot >> %LOG% 2>&1
echo ===== STEP4 FINISHED %date% %time% ===== >> %LOG%
echo Done. You can close this window.
pause
