@echo off
setlocal
title FYP2 Deepfake Detector

REM Keep the lightweight calibration layer from spawning a large OpenMP pool per image.
set "LOKY_MAX_CPU_COUNT=%NUMBER_OF_PROCESSORS%"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"

cd /d "%~dp0"

cls
echo =====================================
echo   FYP2 Deepfake Detector Launcher
echo =====================================
echo Installation all in local venv folder.
echo First run installs required libraries.
echo Later run launch web unless libraries missing.
echo.

set "VENV_DIR=%CD%\venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "TORCH_BUILD=cpu"
set "TORCH_INDEX=cpu"
set "TORCH_CUDA_VERSION="
set "TORCH_VERSION=2.7.1"
set "TORCHVISION_VERSION=0.22.1"

REM -----------------------------
REM Locate system Python
REM -----------------------------
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python is NOT installed or not in PATH.
        echo Please install Python 3.10+ and enable Add to PATH.
        pause
        exit /b 1
    ) else (
        set "PYCALL=py"
    )
) else (
    set "PYCALL=python"
)

REM -----------------------------
REM Create app folders
REM -----------------------------
if not exist "outputs" mkdir "outputs"
if not exist "models" mkdir "models"
if not exist "uploads" mkdir "uploads"

REM -----------------------------
REM Create venv on first run
REM -----------------------------
if not exist "%VENV_PY%" (
    echo Creating virtual environment...
    %PYCALL% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

if not exist "%VENV_PY%" (
    echo ERROR: Virtual environment Python was not found.
    pause
    exit /b 1
)

if not exist "detect_torch_build.py" (
    echo ERROR: detect_torch_build.py not found!
    pause
    exit /b 1
)

REM -----------------------------
REM Install only when needed
REM -----------------------------
set "HAS_NVIDIA=0"
nvidia-smi >nul 2>&1
if not errorlevel 1 set "HAS_NVIDIA=1"

REM Select the best PyTorch wheel for this computer's NVIDIA driver.
REM PyTorch wheels include the CUDA runtime, so the user does not need to install the CUDA Toolkit.
for /f "tokens=1-5 delims=|" %%A in ('%PYCALL% detect_torch_build.py') do (
    set "TORCH_BUILD=%%A"
    set "TORCH_INDEX=%%B"
    set "TORCH_CUDA_VERSION=%%C"
    set "TORCH_VERSION=%%D"
    set "TORCHVISION_VERSION=%%E"
)

set "NEEDS_INSTALL=0"

if "%TORCH_BUILD%"=="cuda" (
    "%VENV_PY%" -c "import sys, torch, torchvision; ok=torch.__version__.startswith('%TORCH_VERSION%') and torchvision.__version__.startswith('%TORCHVISION_VERSION%') and str(torch.version.cuda).startswith('%TORCH_CUDA_VERSION%') and torch.cuda.is_available(); sys.exit(0 if ok else 1)" >nul 2>&1
) else (
    "%VENV_PY%" -c "import sys, torch, torchvision; ok=torch.__version__.startswith('%TORCH_VERSION%') and torchvision.__version__.startswith('%TORCHVISION_VERSION%') and torch.version.cuda is None; sys.exit(0 if ok else 1)" >nul 2>&1
)
if errorlevel 1 set "NEEDS_INSTALL=1"

"%VENV_PY%" -c "import numpy, cv2, PIL, matplotlib, seaborn, sklearn, joblib, flask, werkzeug, safetensors, pandas" >nul 2>&1
if errorlevel 1 set "NEEDS_INSTALL=1"

if "%NEEDS_INSTALL%"=="1" (
    echo Installing or repairing libraries inside venv...
    "%VENV_PY%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo ERROR: Failed to upgrade pip inside venv.
        pause
        exit /b 1
    )

    if "%TORCH_BUILD%"=="cuda" (
        echo NVIDIA GPU detected. Installing PyTorch CUDA %TORCH_CUDA_VERSION% build...
        "%VENV_PY%" -m pip install --force-reinstall torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% --index-url https://download.pytorch.org/whl/%TORCH_INDEX%
    ) else (
        if "%HAS_NVIDIA%"=="1" (
            echo NVIDIA GPU detected, but the installed driver is too old for supported PyTorch CUDA wheels.
            echo Installing CPU PyTorch build. Update the NVIDIA driver to enable GPU acceleration.
        ) else (
            echo No NVIDIA GPU detected. Installing CPU PyTorch build...
        )
        "%VENV_PY%" -m pip install --force-reinstall torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% --index-url https://download.pytorch.org/whl/cpu
    )
    if errorlevel 1 (
        echo ERROR: PyTorch installation failed.
        pause
        exit /b 1
    )

    if "%TORCH_BUILD%"=="cuda" (
        "%VENV_PY%" -c "import sys, torch, torchvision; ok=torch.__version__.startswith('%TORCH_VERSION%') and torchvision.__version__.startswith('%TORCHVISION_VERSION%') and str(torch.version.cuda).startswith('%TORCH_CUDA_VERSION%') and torch.cuda.is_available(); sys.exit(0 if ok else 1)" >nul 2>&1
        if errorlevel 1 (
            echo WARNING: NVIDIA was detected, but the selected CUDA PyTorch build is not usable on this computer.
            echo Falling back to CPU PyTorch build so the app can still run.
            "%VENV_PY%" -m pip install --force-reinstall torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% --index-url https://download.pytorch.org/whl/cpu
            if errorlevel 1 (
                echo ERROR: CPU PyTorch fallback installation failed.
                pause
                exit /b 1
            )
        )
    )

    echo Installing remaining dependencies from requirements.txt...
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already has the correct libraries for this computer.
)

REM -----------------------------
REM Final checks and launch
REM -----------------------------
if not exist "deepfake_detector.py" (
    echo ERROR: deepfake_detector.py not found!
    pause
    exit /b 1
)

"%VENV_PY%" -c "import sys; print('Python:', sys.executable)"
"%VENV_PY%" -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo.
echo =====================================
echo Server Starting...
echo Open: http://127.0.0.1:8080
echo =====================================
echo.

"%VENV_PY%" deepfake_detector.py %*
pause
