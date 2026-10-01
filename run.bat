@echo off
title Moon Phase Auto-Centering Tool
echo Starting Moon Phase Auto-Centering Tool...
echo Open your browser at: http://127.0.0.1:5050
echo.
start "" "http://127.0.0.1:5050"
echo wait...
REM please change path to your python.exe
"Z:\antigravity\venv\Scripts\python.exe" "%~dp0app.py"
pause
