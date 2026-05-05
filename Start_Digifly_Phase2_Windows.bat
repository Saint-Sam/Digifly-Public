@echo off
setlocal

cd /d "%~dp0"

where docker >nul 2>nul
if errorlevel 1 (
    echo Docker was not found. Install Docker Desktop, start it, then run this file again.
    pause
    exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
    echo Docker Desktop is not running yet. Start Docker Desktop, wait until it is ready, then run this file again.
    pause
    exit /b 1
)

docker compose version >nul 2>nul
if errorlevel 1 (
    echo Docker Compose was not found. Update Docker Desktop, then run this file again.
    pause
    exit /b 1
)

set "DIGIFLY_PORT=8888"
set "DIGIFLY_JUPYTER_PORT=%DIGIFLY_PORT%"
set "DIGIFLY_NOTEBOOK_URL=http://localhost:%DIGIFLY_PORT%/lab/tree/START_HERE_Digifly_Phase2.ipynb"

echo Starting Digifly Phase 2 Docker runtime...
docker compose up --build -d phase2-jupyter
if errorlevel 1 (
    echo Docker failed to start the Digifly Phase 2 runtime.
    echo Run "docker compose logs phase2-jupyter" from this folder for details.
    pause
    exit /b 1
)

echo Waiting for JupyterLab at http://localhost:%DIGIFLY_PORT% ...
where curl >nul 2>nul
if errorlevel 1 (
    echo curl was not found, so this launcher will wait briefly and then open the browser.
    timeout /t 15 /nobreak >nul
    goto open_notebook
)

set /a DIGIFLY_TRIES=0

:wait_for_jupyter
curl -fsS "http://localhost:%DIGIFLY_PORT%/api" >nul 2>nul
if not errorlevel 1 goto open_notebook

set /a DIGIFLY_TRIES+=1
if %DIGIFLY_TRIES% GEQ 120 goto open_notebook

timeout /t 2 /nobreak >nul
goto wait_for_jupyter

:open_notebook
echo Opening Digifly one-click notebook...
start "" "%DIGIFLY_NOTEBOOK_URL%"

echo.
echo If the browser did not open, paste this into Chrome or Edge:
echo %DIGIFLY_NOTEBOOK_URL%
echo.
echo In the notebook, run the single code cell. It will open the Phase 2 Workbench through Docker.
echo.
echo To stop Digifly later, run:
echo docker compose down
echo.
pause
