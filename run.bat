@echo off
setlocal
title Nexus BTA

set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\start_nexus.ps1" -ProjectRoot "%~dp0." -StartComfy
if errorlevel 1 (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "& { . '%ROOT%scripts\nexus_terminal.ps1'; Write-NexusLine 'Startup falhou.' 'Error' }"
  pause
  exit /b 1
)

echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { . '%ROOT%scripts\nexus_terminal.ps1'; Write-NexusLine 'Backend e ComfyUI em execucao.' 'Ok'; Write-NexusLine 'Mantenha esta janela aberta enquanto usa a plataforma.' 'Info'; Write-NexusLine 'Pressione qualquer tecla para fechar e parar os servicos.' 'Info' }"
pause >nul

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\stop_nexus.ps1" -ProjectRoot "%~dp0."

endlocal
