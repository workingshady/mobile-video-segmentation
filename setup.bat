@echo off
REM One-time setup: conda env "mvseg" + PyTorch (CUDA 12.4) + editor deps + EfficientTAM + model weights.
REM Run from an Anaconda Prompt (or any shell where `conda` works).
setlocal
cd /d "%~dp0"

where conda >nul 2>nul || (echo conda not found. Install Anaconda/Miniconda first. & exit /b 1)

conda env list | findstr /b /c:"mvseg " >nul || conda create -n mvseg python=3.11 -y || exit /b 1

echo [1/4] PyTorch with CUDA 12.4 (use the CPU build if you have no NVIDIA GPU)
conda run -n mvseg pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 || exit /b 1

echo [2/4] Editor + model libraries
conda run -n mvseg pip install -r requirements.txt || exit /b 1

echo [3/4] EfficientTAM code (third_party/EfficientTAM)
if not exist third_party\EfficientTAM git clone --depth 1 https://github.com/yformer/EfficientTAM.git third_party\EfficientTAM
set Efficient_Track_Anything_BUILD_CUDA=0
pushd third_party\EfficientTAM
conda run -n mvseg pip install --no-build-isolation --no-deps -e . || (popd & exit /b 1)
popd

echo [4/4] Model weights (~6 GB; re-run any time, finished files are skipped)
conda run -n mvseg python scripts\download_models.py

echo.
echo Done. Start the editor with: run_editor.bat
endlocal
