@echo off
title Deepfake Analysis Launcher

REM Check if virtual environment exists
if not exist venv (
    cls
    echo =====================================
    echo              DISCLAIMER
    echo =====================================
    echo This is a student final year project.
    echo You can feel free at ease, no virus,
    echo no worries that the pip install will
    echo be globally installed because all this
    echo will be installed on a virtual environment
    echo and can be easily removed by deleting
    echo the entire file root.
    echo.
    echo This whole entire files are around 1GB
    echo and only need internet for first time
    echo running since we need to install the
    echo python library dependencies.
    echo =====================================
    echo.
    echo The installer will continue in 30 seconds...
    echo Press any key to continue immediately
    
    REM Wait 30 seconds or until key press
    timeout /t 30 >nul
)

cls
echo =====================================
echo      Deepfake Images Detector
echo =====================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python is NOT installed or not in PATH.
        echo Please install Python 3.10+ and enable "Add to PATH".
        pause
        exit /b
    ) else (
        set pycall=py
    )
) else (
    set pycall=python
)

REM Create venv if not exists
if not exist venv (
    echo Creating virtual environment...
    %pycall% -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b
    )
    echo Virtual environment created successfully.
    echo.
) else (
    echo Virtual environment detected.
    echo.
)

REM Activate venv
call venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b
)

REM Install dependencies on first run only
if not exist venv\Lib\site-packages\torch\__init__.py (
    echo First run detected. Installing dependencies...
    echo.
    
    echo Updating pip...
    python -m pip install --upgrade pip
    
    if not exist requirements.txt (
        echo ERROR: requirements.txt not found.
        pause
        exit /b
    )
    
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies. Something went wrong ":("
        pause
        exit /b
    )
    echo Dependencies installed successfully!
    echo.
)

REM Launch Web Server
echo =====================================
echo     Server Starting...
echo     Open: http://127.0.0.1:8080
echo =====================================
echo.

if not exist deepfake_detector.py (
    echo ERROR: deepfake_detector.py not found!
    pause
    exit /b
)

python deepfake_detector.py %*

pause
