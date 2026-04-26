@echo off
REM ============================================================================
REM  FaceFusion - Run launcher (Windows)
REM ----------------------------------------------------------------------------
REM  Activate venv (.venv) va goi facefusion.py.
REM
REM  Mac dinh: chay UI voi preset chat luong cao (facefusion.high-quality.ini).
REM
REM  Vi du:
REM      run.bat                                   (UI, high-quality preset)
REM      run.bat --quality fast                    (UI, ini mac dinh)
REM      run.bat headless-run -s a.jpg -t b.mp4 -o c.mp4
REM      run.bat batch-run --source-pattern "src/*.jpg" --target-pattern "tgt/*.mp4" --output-pattern "out/{target_name}.mp4"
REM
REM  Bat ki tham so nao khac --quality deu duoc forward thang cho facefusion.py.
REM ============================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"

REM ---- Quality preset ----
set "QUALITY=high"
set "ARGS="

:parse_args
if "%~1"=="" goto :after_args
if /I "%~1"=="--quality" (
    set "QUALITY=%~2"
    shift
    shift
    goto :parse_args
)
set "ARGS=!ARGS! %1"
shift
goto :parse_args
:after_args

REM ---- Activate venv ----
if not exist ".venv\Scripts\activate.bat" (
    echo [error] .venv chua duoc tao. Chay install.bat truoc.
    exit /b 1
)
call ".venv\Scripts\activate.bat"

REM ---- Chon config theo quality ----
REM Preset alias:
REM   high     -> facefusion.high-quality.ini  (max realism, >=12 GB VRAM)
REM   balanced -> facefusion.balanced.ini      (clip ngan, ~6-8 GB VRAM)
REM   fast     -> facefusion.fast.ini          (real-time / preview)
REM   default  -> facefusion.ini               (config goc cua repo)
REM   <name>   -> facefusion.<name>.ini        (user-defined preset)
set "CONFIG_FLAG="
if /I "%QUALITY%"=="high" (
    if exist "facefusion.high-quality.ini" (
        set "CONFIG_FLAG=--config-path facefusion.high-quality.ini"
        echo [info] Quality preset: HIGH ^(facefusion.high-quality.ini^)
    ) else (
        echo [warn] Khong thay facefusion.high-quality.ini, dung config mac dinh.
    )
) else if /I "%QUALITY%"=="balanced" (
    if exist "facefusion.balanced.ini" (
        set "CONFIG_FLAG=--config-path facefusion.balanced.ini"
        echo [info] Quality preset: BALANCED ^(facefusion.balanced.ini^)
    ) else (
        echo [warn] Khong thay facefusion.balanced.ini, dung config mac dinh.
    )
) else if /I "%QUALITY%"=="fast" (
    if exist "facefusion.fast.ini" (
        set "CONFIG_FLAG=--config-path facefusion.fast.ini"
        echo [info] Quality preset: FAST ^(facefusion.fast.ini^)
    ) else (
        echo [info] Quality preset: FAST ^(facefusion.ini mac dinh^)
    )
) else if /I "%QUALITY%"=="default" (
    echo [info] Quality preset: DEFAULT ^(facefusion.ini mac dinh^)
) else (
    if exist "facefusion.%QUALITY%.ini" (
        set "CONFIG_FLAG=--config-path facefusion.%QUALITY%.ini"
        echo [info] Quality preset: %QUALITY% ^(facefusion.%QUALITY%.ini^)
    ) else (
        echo [warn] Khong tim thay facefusion.%QUALITY%.ini, dung mac dinh.
    )
)

REM ---- Default subcommand: run ----
set "TRIMMED=!ARGS: =!"
set "SUBCMD=run"
set "FIRST_ARG="
for /f "tokens=1" %%a in ("!ARGS!") do set "FIRST_ARG=%%a"
if not "!FIRST_ARG!"=="" (
    REM Neu user da truyen subcommand thi giu nguyen
    set "SUBCMD=!FIRST_ARG!"
    set "REST=!ARGS:*%FIRST_ARG%=!"
) else (
    set "REST="
)

if "!FIRST_ARG!"=="" (
    REM Khong co arg -> dung 'run' va apply config
    python facefusion.py run %CONFIG_FLAG%
    exit /b !errorlevel!
)

REM Co arg -> forward nguyen, them config-path neu chua co
echo !REST! | findstr /C:"--config-path" >nul
if errorlevel 1 (
    python facefusion.py !SUBCMD! %CONFIG_FLAG%!REST!
) else (
    python facefusion.py !SUBCMD!!REST!
)
exit /b %errorlevel%
