@echo off
setlocal
title Nexus BTA LAN

set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\start_network.ps1" -ProjectRoot "%~dp0." -Mode lan
if errorlevel 1 (
  echo.
  echo Nexus LAN startup failed.
  pause
  exit /b 1
)

endlocal
