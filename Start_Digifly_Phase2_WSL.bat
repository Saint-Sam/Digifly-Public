@echo off
setlocal

cd /d "%~dp0"

where wsl.exe >nul 2>nul
if errorlevel 1 (
    echo WSL was not found.
    echo Install WSL with Ubuntu, restart this terminal if needed, then run this file again.
    echo.
    echo Suggested command from PowerShell:
    echo wsl --install -d Ubuntu
    echo.
    pause
    exit /b 1
)

wsl.exe --status >nul 2>nul
if errorlevel 1 (
    echo WSL is installed, but no default Linux distro appears to be ready yet.
    echo Open Ubuntu once from the Start Menu, finish its setup, then run this file again.
    echo.
    pause
    exit /b 1
)

if defined DIGIFLY_WSL_DISTRO (
    echo Starting Digifly Phase 2 WSL runtime in %DIGIFLY_WSL_DISTRO%...
    wsl.exe -d "%DIGIFLY_WSL_DISTRO%" --cd "%CD%" bash ./Start_Digifly_Phase2_WSL.sh %*
) else (
    echo Starting Digifly Phase 2 WSL runtime in the default WSL distro...
    wsl.exe --cd "%CD%" bash ./Start_Digifly_Phase2_WSL.sh %*
)

if errorlevel 1 (
    echo.
    echo Digifly WSL startup failed.
    echo Review the messages above, then run this file again after fixing the issue.
    echo.
    pause
    exit /b 1
)
