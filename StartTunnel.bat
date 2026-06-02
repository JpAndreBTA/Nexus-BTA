@echo off
setlocal
title Nexus BTA Tunnel

set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\start_network.ps1" -ProjectRoot "%~dp0." -Mode tunnel
if errorlevel 1 (
  echo.
  echo Nexus tunnel startup failed.
  pause
  exit /b 1
)

endlocal
