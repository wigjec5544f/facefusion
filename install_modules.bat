@echo off
REM ============================================================================
REM  FaceFusion - Install Python Modules
REM ----------------------------------------------------------------------------
REM  Cai cac Python module can thiet vao venv hien tai.
REM
REM  Usage:
REM      install_modules.bat              (auto-detect: cuda neu co NVIDIA, nguoc lai default)
REM      install_modules.bat cuda         (NVIDIA + CUDA)
REM      install_modules.bat directml     (Windows DirectML - AMD/Intel/NVIDIA)
REM      install_modules.bat openvino     (Intel OpenVINO)
REM      install_modules.bat qnn          (Snapdragon NPU)
REM      install_modules.bat default      (CPU only)
REM
REM  Yeu cau: da co Python 3.12+ trong PATH va venv da activate
REM           (chay thong qua install.bat de tu dong)
REM ============================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"

REM ---- 1. Lay onnxruntime variant ----
set "ORT_VARIANT=%~1"

if "%ORT_VARIANT%"=="" (
    echo [info] Khong chi dinh onnxruntime variant, dang auto-detect GPU...
    where nvidia-smi >nul 2>&1
    if !errorlevel! == 0 (
        nvidia-smi >nul 2>&1
        if !errorlevel! == 0 (
            set "ORT_VARIANT=cuda"
            echo [info] Phat hien NVIDIA GPU -^> dung CUDA
        )
    )
    if "!ORT_VARIANT!"=="" (
        set "ORT_VARIANT=directml"
        echo [info] Khong phat hien NVIDIA -^> dung DirectML ^(chay tren AMD/Intel/NVIDIA Windows^)
    )
)

echo [info] onnxruntime variant: %ORT_VARIANT%

REM ---- 2. Kiem tra Python ----
where python >nul 2>&1
if errorlevel 1 (
    echo [error] Khong tim thay python trong PATH.
    echo         Cai Python 3.12+ tu https://www.python.org/downloads/ va chon "Add to PATH".
    exit /b 1
)

REM ---- 3. Upgrade pip + wheel ----
echo [info] Nang cap pip / setuptools / wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :pip_fail

REM ---- 4. Cai dependencies + onnxruntime ----
echo [info] Chay installer cua FaceFusion...
python install.py --onnxruntime %ORT_VARIANT% --skip-conda
if errorlevel 1 goto :pip_fail

echo.
echo [done] Da cai dat xong cac module Python.
echo        De chay app:    run.bat
exit /b 0

:pip_fail
echo [error] pip / installer that bai. Kiem tra log o tren.
exit /b 1
