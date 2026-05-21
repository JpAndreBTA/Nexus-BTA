@echo off
setlocal

set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\update_nexus.ps1" -ProjectRoot "%~dp0."
if errorlevel 1 (
  echo [NEXUS BTA] Update failed.
  pause
  exit /b 1
)

echo.
echo [NEXUS BTA] Update complete.
pause
endlocal
