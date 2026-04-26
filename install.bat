@echo off
REM ============================================================================
REM  FaceFusion - One-shot Installer (Windows)
REM ----------------------------------------------------------------------------
REM  Bootstrap may MOI:
REM    1. Kiem tra Python 3.10-3.13 (cai bang winget neu thieu)
REM    2. Kiem tra git (canh bao neu thieu)
REM    3. Cai ffmpeg qua winget (neu chua co)
REM    4. Tao venv .venv tai thu muc repo
REM    5. Goi install_modules.bat de cai requirements + onnxruntime
REM    6. Force-download tat ca model (offline-ready sau khi cai xong)
REM
REM  Tham so tuy chon:
REM      install.bat                  (auto-detect onnxruntime variant)
REM      install.bat cuda             (ep dung CUDA)
REM      install.bat directml         (ep dung DirectML)
REM      install.bat default          (CPU only)
REM      install.bat openvino|qnn|rocm|migraphx
REM
REM  Sau khi cai xong, dung run.bat de mo UI.
REM ============================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"

set "ORT_VARIANT=%~1"

echo ============================================================
echo  FaceFusion installer - %DATE% %TIME%
echo  Repo dir: %CD%
echo ============================================================

REM ---- 1. Python ----
echo.
echo [step 1/6] Kiem tra Python...
where python >nul 2>&1
if errorlevel 1 (
    echo [warn] Khong co python trong PATH. Thu cai bang winget...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo [error] Khong co winget. Cai Python 3.12 thu cong tu:
        echo         https://www.python.org/downloads/
        echo         Nho tick "Add Python to PATH" khi cai.
        exit /b 1
    )
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo [error] winget khong cai duoc Python.
        exit /b 1
    )
    echo [info] Da cai Python. Vui long DONG va MO LAI cmd.exe roi chay lai install.bat.
    exit /b 0
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo [info] Python: !PYVER!

REM ---- 2. Git (canh bao neu thieu, khong bat buoc) ----
echo.
echo [step 2/6] Kiem tra git...
where git >nul 2>&1
if errorlevel 1 (
    echo [warn] Khong tim thay git. Khong bat buoc, nhung nen cai de pull update:
    echo        winget install -e --id Git.Git
) else (
    echo [info] git: OK
)

REM ---- 3. ffmpeg ----
echo.
echo [step 3/6] Kiem tra ffmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [warn] Khong co ffmpeg. Thu cai bang winget...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo [error] Khong co winget. Tai ffmpeg thu cong tu https://www.gyan.dev/ffmpeg/builds/
        echo         va them vao PATH, sau do chay lai install.bat.
        exit /b 1
    )
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo [error] winget khong cai duoc ffmpeg.
        exit /b 1
    )
    echo [info] Da cai ffmpeg. Co the can mo lai cmd.exe de PATH cap nhat.
) else (
    echo [info] ffmpeg: OK
)

REM ---- 4. venv ----
echo.
echo [step 4/6] Tao virtual environment .venv ...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 (
        echo [error] Khong tao duoc venv. Kiem tra quyen ghi va dung luong dia.
        exit /b 1
    )
) else (
    echo [info] .venv da ton tai, bo qua tao moi.
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [error] Khong activate duoc .venv.
    exit /b 1
)
echo [info] Active venv: %VIRTUAL_ENV%

REM ---- 5. Cai modules ----
echo.
echo [step 5/6] Cai Python modules...
call install_modules.bat %ORT_VARIANT%
if errorlevel 1 (
    echo [error] install_modules.bat that bai.
    exit /b 1
)

REM ---- 6. Force-download models ----
echo.
echo [step 6/6] Tai truoc cac model AI ^(co the mat vai phut^)...
python facefusion.py force-download
if errorlevel 1 (
    echo [warn] Tai model bi loi - co the do mang. App van chay duoc, model se tai khi can.
)

echo.
echo ============================================================
echo  HOAN TAT
echo  - Chay UI:               run.bat
echo  - Chay UI ^(che do nhanh^): run.bat --quality fast
echo  - Headless / CLI:         run.bat headless-run -s src.jpg -t tgt.mp4 -o out.mp4
echo ============================================================
exit /b 0
