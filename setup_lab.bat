@echo off
chcp 65001 >nul
echo ============================================
echo   OpenVINO Workshop 环境安装
echo ============================================
echo.

:: 检查 Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10 或更高版本。
    pause
    exit /b 1
)

:: 创建虚拟环境
echo [1/5] 创建虚拟环境 ov_workshop ...
if not exist "ov_workshop" (
    python -m venv ov_workshop
    echo      虚拟环境创建完成
) else (
    echo      虚拟环境已存在，跳过创建
)

:: 激活虚拟环境
echo [2/5] 激活虚拟环境 ...
call ov_workshop\Scripts\activate.bat

:: 升级 pip
echo [3/5] 升级 pip ...
python -m pip install --upgrade pip

:: 安装依赖
echo [4/5] 安装依赖 (可能需要几分钟) ...
pip install -r requirements.txt

:: 克隆并安装 Qwen3-ASR 和 Qwen3-TTS
echo [5/5] 安装 Qwen3-ASR / Qwen3-TTS 推理库 ...

:: Qwen3-ASR
if not exist "lab2-speech-recognition\Qwen3-ASR" (
    echo      克隆 Qwen3-ASR ...
    cd lab2-speech-recognition
    git clone https://github.com/QwenLM/Qwen3-ASR.git
    cd Qwen3-ASR
    git checkout c17a131fe028b2e428b6e80a33d30bb4fa57b8df
    cd ..
    pip install -q -e Qwen3-ASR
    cd ..
) else (
    echo      Qwen3-ASR 已存在，跳过克隆
    pip install -q -e lab2-speech-recognition\Qwen3-ASR
)

:: Qwen3-TTS
if not exist "lab3-text-to-speech\Qwen3-TTS" (
    echo      克隆 Qwen3-TTS ...
    cd lab3-text-to-speech
    git clone https://github.com/QwenLM/Qwen3-TTS.git
    cd Qwen3-TTS
    git checkout 1ab0dd75353392f28a0d05d9ca960c9954b13c83
    cd ..
    pip install -q -e Qwen3-TTS
    cd ..
) else (
    echo      Qwen3-TTS 已存在，跳过克隆
    pip install -q -e lab3-text-to-speech\Qwen3-TTS
)

echo.
echo ============================================
echo   安装完成！请运行 run_lab.bat 启动 JupyterLab
echo ============================================
pause
