@echo off
setlocal

set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\start_nexus.ps1" -ProjectRoot "%~dp0."
if errorlevel 1 (
  echo [NEXUS BTA] Startup failed.
  pause
  exit /b 1
)

echo.
echo [NEXUS BTA] Backend and embedded ComfyUI are running.
echo [NEXUS BTA] Keep this window open while using the app.
echo [NEXUS BTA] Press any key to close the app and stop backend/frontend services.
pause >nul

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\stop_nexus.ps1" -ProjectRoot "%~dp0."

endlocal
