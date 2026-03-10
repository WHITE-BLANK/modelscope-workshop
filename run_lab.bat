@echo off
chcp 65001 >nul
echo ============================================
echo   启动 OpenVINO Workshop
echo ============================================
echo.

:: 激活虚拟环境
call ov_workshop\Scripts\activate.bat

:: 启动 JupyterLab
echo 正在启动 JupyterLab ...
jupyter lab .

:: 退出
call deactivate
pause
