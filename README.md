# OpenVINO AI 应用实战 Workshop

本 Workshop 基于 [OpenVINO™](https://github.com/openvinotoolkit/openvino) 工具套件，引导你完成四个动手实验（Lab），涵盖视觉语言理解、语音识别、语音合成和文生图四大 AI 应用场景。所有模型均已预先导出为 OpenVINO IR 格式，你只需下载模型并运行推理即可。

## 💻 系统要求

- **操作系统**: Windows 11
- **Python**: 3.10+
- **GPU（可选）**: Intel® Arc™ 系列独立显卡 或 Intel® Core™ Ultra 集成显卡

## 📖 安装步骤

### 1. 创建虚拟环境并安装依赖

双击运行 `setup_lab.bat`，脚本会自动：
1. 使用 Python `venv` 创建虚拟环境 `ov_workshop`
2. 安装 `requirements.txt` 中的所有依赖
3. 克隆并安装 Qwen3-ASR / Qwen3-TTS 推理所需的代码库

### 2. 启动 JupyterLab

双击运行 `run_lab.bat`，脚本会：
1. 激活虚拟环境
2. 启动 JupyterLab，浏览器会自动打开

## 🧪 实验列表

| 实验 | 主题 | 模型 | 简介 | 链接 |
|------|------|------|------|------|
| Lab 1 | 多模态视觉语言理解 | Qwen3-VL-4B | 用 OpenVINO 运行视觉语言大模型，支持图像理解与多轮对话 | [进入](lab1-multimodal-vlm/) |
| Lab 2 | 语音识别 (ASR) | Qwen3-ASR-0.6B | 用 OpenVINO 运行语音识别模型，支持 52+ 种语言 | [进入](lab2-speech-recognition/) |
| Lab 3 | 语音合成 (TTS) | Qwen3-TTS-0.6B | 用 OpenVINO 运行语音合成模型，支持多语言多说话人 | [进入](lab3-text-to-speech/) |
| Lab 4 | 文生图 | Z-Image-Turbo | 用 OpenVINO 运行文生图模型，支持中英双语和高质量图像生成 | [进入](lab4-image-generation/) |

## 📁 项目结构

```
openvino-workshop/
├── README.md                     # 本文件
├── requirements.txt              # 统一依赖
├── setup_lab.bat                 # 环境安装脚本
├── run_lab.bat                   # 启动 JupyterLab
├── lab1-multimodal-vlm/          # Lab 1: Qwen3-VL 多模态视觉语言
├── lab2-speech-recognition/      # Lab 2: Qwen3-ASR 语音识别
├── lab3-text-to-speech/          # Lab 3: Qwen3-TTS 语音合成
└── lab4-image-generation/        # Lab 4: Z-Image-Turbo 文生图
```

## 🔗 参考资源

- [OpenVINO 官方文档](https://docs.openvino.ai/)
- [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
- [OpenVINO DevCon Workshop](https://github.com/openvino-dev-samples/openvino-devcon-workshop)
