@echo off
REM One-click entry point (Windows): fetch the datasets if they are not
REM there yet, then build and run the experiment suite.
REM
REM If the download fails (no internet, firewall), the program still runs
REM on a deterministic synthetic fallback and prints that it is doing so.

cd /d "%~dp0"

if not exist "data\mnist\train-images-idx3-ubyte" (
    echo Fetching datasets ^(requires internet, one time only^)...
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\download_data.ps1"
    if errorlevel 1 (
        echo.
        echo Download failed - continuing with the synthetic fallback.
        echo.
    )
)

cargo run --release

echo.
pause
