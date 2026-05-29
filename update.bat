@echo off
setlocal
title Nexus BTA Update

set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\configure_model_paths.ps1" -ProjectRoot "%~dp0." -PromptDownloads
if errorlevel 1 (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "& { . '%ROOT%scripts\nexus_terminal.ps1'; Write-NexusLine 'Model path setup failed.' 'Error' }"
  pause
  exit /b 1
)
if exist "%ROOT%config\nexus_startup_env.cmd" call "%ROOT%config\nexus_startup_env.cmd"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\update_nexus.ps1" -ProjectRoot "%~dp0."
if errorlevel 1 (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "& { . '%ROOT%scripts\nexus_terminal.ps1'; Write-NexusLine 'Update falhou.' 'Error' }"
  pause
  exit /b 1
)

echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { . '%ROOT%scripts\nexus_terminal.ps1'; Write-NexusLine 'Update concluido.' 'Ok' }"
pause
endlocal
