@echo off
REM Start Track Studio at http://127.0.0.1:7860 (opens your browser).
REM Uses .venv (light venv on top of an existing PyTorch env) if present, else the "mvseg" conda env from setup.bat.
cd /d "%~dp0"
set USE_TF=0
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -m editor %*
) else (
  conda run --no-capture-output -n mvseg python -m editor %*
)
