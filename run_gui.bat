@echo off
title Binaural Simulation v3.0

:menu
cls
echo.
echo =============================================
echo   Binaural Simulation v3.0
echo =============================================
echo.
echo   [1] Run Simulation    (Python)
echo   [2] Open GUI           (MATLAB 3D Visual)
echo   [0] Exit
echo.
set /p choice="Select: "

if "%choice%"=="1" goto run_sim
if "%choice%"=="2" goto launch_gui
if "%choice%"=="0" exit /b 0
echo Invalid choice.
pause
goto menu

:run_sim
echo.
echo Starting Python simulation...
python "%~dp0run.py"
echo.
echo Press any key to return to menu...
pause >nul
goto menu

:launch_gui
echo.
echo Looking for MATLAB...

set MATLAB=

where matlab >nul 2>&1
if %errorlevel% equ 0 (
    for /f "delims=" %%i in ('where matlab 2^>nul') do set MATLAB=%%i
)

if "%MATLAB%"=="" (
    for /d %%d in ("C:\Program Files\MATLAB\R*") do (
        if exist "%%d\bin\matlab.exe" set MATLAB=%%d\bin\matlab.exe
    )
)

if "%MATLAB%"=="" (
    for /d %%d in ("C:\Program Files (x86)\MATLAB\R*") do (
        if exist "%%d\bin\matlab.exe" set MATLAB=%%d\bin\matlab.exe
    )
)

if "%MATLAB%"=="" (
    echo.
    echo MATLAB not found. Manual steps:
    echo   1. Open MATLAB
    echo   2. cd D:\shengxuedingwei2\matlab
    echo   3. Run: binaural_gui
    echo.
    pause
    goto menu
)

echo Found: %MATLAB%
echo Launching GUI...

set PROJ_DIR=%~dp0
set PROJ_DIR=%PROJ_DIR:\=/%

start "BinauralGUI" "%MATLAB%" -nosplash -r "addpath('%PROJ_DIR%matlab'); binaural_gui();"

echo.
echo GUI launched - switch to the GUI window.
pause
goto menu
