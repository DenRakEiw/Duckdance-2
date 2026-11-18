@echo off
rem Startet Duck Retarget ohne Konsolenfenster.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" "app\main.py"
