@echo off
setlocal
REM ============================================================
REM  AML-Sec Security Evaluation Framework - one-click launcher
REM  Runs the full real-data pipeline end to end on this machine,
REM  then opens the dashboard. Nothing simulated: every number is
REM  measured live. Full run takes roughly 40-50 minutes on CPU.
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.x first.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  AML-SEC :: full evaluation pipeline (real runs, no fake)
echo ============================================================
echo.

REM ---- 1. six attack modules -----------------------------------
for %%E in (run_poisoning run_backdoor run_extraction run_membership run_membership_small run_adversarial run_supply_chain) do (
    echo [step] experiments\%%E.py
    python experiments\%%E.py
    if errorlevel 1 goto :fail
)

REM ---- 2. cross-threat matrix ----------------------------------
echo [step] experiments\run_cross_threat.py   ^(5 regimes x 3 scenarios x 3 seeds^)
python experiments\run_cross_threat.py
if errorlevel 1 goto :fail

echo [step] experiments\run_cross_srcacc.py   ^(clean source-class accuracy^)
python experiments\run_cross_srcacc.py
if errorlevel 1 goto :fail

echo [step] experiments\run_cross_matrix.py   ^(paired dS + classification^)
python experiments\run_cross_matrix.py
if errorlevel 1 goto :fail

REM ---- 3. unified profile + final artifacts --------------------
echo [step] experiments\run_final.py          ^(unified profile + Pareto^)
python experiments\run_final.py
if errorlevel 1 goto :fail

echo [step] experiments\build_report.py
python experiments\build_report.py
if errorlevel 1 goto :fail

echo [step] scripts\build_dashboard.py
python scripts\build_dashboard.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  DONE. All results in results\  ^(JSON + REPORT.md^)
echo  Opening dashboard...
echo ============================================================
start "" "%~dp0dashboard.html"
pause
exit /b 0

:fail
echo.
echo [ERROR] A step failed - see the output above. Nothing was faked;
echo         fix the failing step and re-run.
pause
exit /b 1