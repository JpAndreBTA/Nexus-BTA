@echo off
setlocal
title Nexus BTA

set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\configure_model_paths.ps1" -ProjectRoot "%~dp0."
if errorlevel 1 (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "& { . '%ROOT%scripts\nexus_terminal.ps1'; Write-NexusLine 'Model path setup failed.' 'Error' }"
  pause
  exit /b 1
)
if exist "%ROOT%config\nexus_startup_env.cmd" call "%ROOT%config\nexus_startup_env.cmd"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\start_nexus.ps1" -ProjectRoot "%~dp0." -StartComfy -ComfyWarmupSeconds 75
if errorlevel 1 (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "& { . '%ROOT%scripts\nexus_terminal.ps1'; Write-NexusLine 'Startup failed.' 'Error' }"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "& { . '%ROOT%scripts\nexus_terminal.ps1'; Write-NexusLine 'Open update.bat to repair or refresh dependencies, then run run.bat again.' 'Warn' }"
  pause
  exit /b 1
)

echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { . '%ROOT%scripts\nexus_terminal.ps1'; Write-NexusLine 'Backend is running. ComfyUI starts in the background or on first generation.' 'Ok'; Write-NexusLine 'Keep this window open while using the platform.' 'Info'; Write-NexusLine 'Press any key to close and stop services.' 'Info' }"
pause >nul

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\stop_nexus.ps1" -ProjectRoot "%~dp0."

endlocal
