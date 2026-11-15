@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\open_football_picks.ps1"
if errorlevel 1 pause
