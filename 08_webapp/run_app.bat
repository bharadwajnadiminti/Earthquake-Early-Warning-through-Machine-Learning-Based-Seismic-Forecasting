@echo off
REM Launches the stage 08 forecast dashboard (see README.md).
REM Hardcoded absolute path so this .bat works no matter where it's run
REM from - a terminal in another folder, a Desktop/Start Menu shortcut
REM copied elsewhere, double-clicked from Explorer, etc.
REM Uses `py -3.14` specifically - the plain `python` on this machine
REM resolves to an older Anaconda install without xgboost installed.

cd /d "C:\Users\bhara\Desktop\USELESS\project\scripts\08_webapp"

echo Starting Earthquake Forecaster dashboard...
echo First load takes ~15-20s (computing features for all active cells).
echo.

py -3.14 app.py

echo.
echo Server stopped.
pause
