"""Gradio UI for the pronunciation correction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import openvino as ov

from pronunciation_correction_pipeline import PronunciationCorrectionPipeline, QwenAudioOVCorrector


PIPELINE_CACHE: Dict[Tuple[str, str, str, str], PronunciationCorrectionPipeline] = {}
TTS_ONLY_PIPELINE_CACHE: Dict[Tuple[str, str, str], PronunciationCorrectionPipeline] = {}


def _resolve_ov_device(requested_device: str) -> Tuple[str, str]:
    """Resolve requested device to a robust OpenVINO device string."""
    req = (requested_device or "CPU").strip().upper()
    core = ov.Core()
    available = set(core.available_devices)

    if req == "GPU":
        if "GPU" in available:
            return "AUTO:GPU,CPU", "已启用 AUTO:GPU,CPU（优先 GPU，失败自动回退 CPU）。"
        return "CPU", "未检测到 OpenVINO 可用的 Intel GPU，已自动切换为 CPU。"

    if req == "NPU":
        if "NPU" in available:
            return "AUTO:NPU,CPU", "已启用 AUTO:NPU,CPU（优先 NPU，失败自动回退 CPU）。"
        return "CPU", "未检测到 OpenVINO 可用的 NPU，已自动切换为 CPU。"

    return "CPU", "使用 CPU 推理。"


def _is_gpu_context_error(exc: Exception) -> bool:
    """Check whether exception matches common OpenVINO GPU context init failures."""
    msg = str(exc)
    patterns = [
        "Context was not initialized",
        "intel_gpu",
        "[GPU]",
    ]
    return any(p in msg for p in patterns)


def _get_or_create_pipeline(
    qwen_audio_model: str,
    tts_model_dir: str,
    device: str,
    speaker: str,
) -> PronunciationCorrectionPipeline:
    """Reuse loaded models across requests to reduce startup time."""
    resolved_device, _ = _resolve_ov_device(device)
    key = (qwen_audio_model.strip(), tts_model_dir.strip(), resolved_device, speaker.strip().lower())
    if key not in PIPELINE_CACHE:
        project_root = Path(__file__).resolve().parent
        PIPELINE_CACHE[key] = PronunciationCorrectionPipeline(
            project_root=str(project_root),
            tts_model_dir=tts_model_dir.strip(),
            qwen_audio_model_dir=qwen_audio_model.strip(),
            device=resolved_device,
            tts_speaker=speaker.strip().lower(),
        )
    return PIPELINE_CACHE[key]


def _get_or_create_tts_pipeline(
    tts_model_dir: str,
    device: str,
    speaker: str,
) -> PronunciationCorrectionPipeline:
    """Create TTS-only pipeline for reference-audio generation."""
    resolved_device, _ = _resolve_ov_device(device)
    key = (tts_model_dir.strip(), resolved_device, speaker.strip().lower())
    if key not in TTS_ONLY_PIPELINE_CACHE:
        project_root = Path(__file__).resolve().parent
        TTS_ONLY_PIPELINE_CACHE[key] = PronunciationCorrectionPipeline(
            project_root=str(project_root),
            tts_model_dir=tts_model_dir.strip(),
            qwen_audio_model_dir=None,
            device=resolved_device,
            tts_speaker=speaker.strip().lower(),
        )
    return TTS_ONLY_PIPELINE_CACHE[key]


def build_default_prompt(text: str) -> str:
    """Build default correction prompt from current text."""
    current_text = (text or "").strip()
    if not current_text:
        current_text = "[请在左侧输入待练习文本]"
    return QwenAudioOVCorrector.build_correction_prompt(current_text)


def generate_reference_audio_only(
    text: str,
    tts_model_dir: str,
    device: str,
    language: str,
    speaker: str,
):
    """Generate only reference audio from input text."""
    if not text or not text.strip():
        return None, "请输入待练习文本后再生成示范音频。"
    if not tts_model_dir or not tts_model_dir.strip():
        return None, "请填写 TTS OpenVINO 模型目录。"

    try:
        _, device_msg = _resolve_ov_device(device)
        tts_pipeline = _get_or_create_tts_pipeline(
            tts_model_dir=tts_model_dir,
            device=device,
            speaker=speaker,
        )
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        reference_audio_out = output_dir / "reference_demo.wav"
        lang = None if language == "Auto" else language
        audio_path = tts_pipeline.generate_reference_audio(
            text=text.strip(),
            output_wav_path=str(reference_audio_out),
            language=lang,
        )
        return audio_path, f"示范音频已生成。{device_msg}"
    except Exception as e:
        if _is_gpu_context_error(e):
            try:
                # GPU 上下文初始化失败时自动回退 CPU
                tts_pipeline = _get_or_create_tts_pipeline(
                    tts_model_dir=tts_model_dir,
                    device="CPU",
                    speaker=speaker,
                )
                output_dir = Path("outputs")
                output_dir.mkdir(parents=True, exist_ok=True)
                reference_audio_out = output_dir / "reference_demo.wav"
                lang = None if language == "Auto" else language
                audio_path = tts_pipeline.generate_reference_audio(
                    text=text.strip(),
                    output_wav_path=str(reference_audio_out),
                    language=lang,
                )
                return audio_path, "GPU 初始化失败，已自动回退到 CPU 并成功生成示范音频。"
            except Exception as e2:
                return None, f"示范音频生成失败（GPU 回退 CPU 也失败）: {type(e2).__name__}: {e2}"
        return None, f"示范音频生成失败: {type(e).__name__}: {e}"


def run_pronunciation_correction(
    text: str,
    user_audio_path: str,
    correction_prompt: str,
    qwen_audio_model: str,
    tts_model_dir: str,
    device: str,
    language: str,
    speaker: str,
):
    """Execute end-to-end pipeline and return UI-friendly outputs."""
    if not text or not text.strip():
        return None, "", "请输入待练习文本。"
    if not user_audio_path:
        return None, "", "请上传或录制跟读音频。"
    if not qwen_audio_model or not qwen_audio_model.strip():
        return None, "", "请填写 Qwen-Audio 模型路径或模型 ID。"
    if not tts_model_dir or not tts_model_dir.strip():
        return None, "", "请填写 TTS OpenVINO 模型目录。"

    try:
        pipeline = _get_or_create_pipeline(
            qwen_audio_model=qwen_audio_model,
            tts_model_dir=tts_model_dir,
            device=device,
            speaker=speaker,
        )

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        reference_audio_out = output_dir / "reference_demo.wav"

        lang = None if language == "Auto" else language
        result = pipeline.run(
            original_text=text.strip(),
            user_audio_path=user_audio_path,
            reference_audio_out=str(reference_audio_out),
            language=lang,
            correction_prompt=correction_prompt,
        )
        return result.reference_audio_path, result.correction_prompt, result.correction_text
    except Exception as e:
        if _is_gpu_context_error(e):
            try:
                # Qwen-Audio 或 TTS 在 GPU 初始化失败时，自动回退 CPU。
                pipeline = _get_or_create_pipeline(
                    qwen_audio_model=qwen_audio_model,
                    tts_model_dir=tts_model_dir,
                    device="CPU",
                    speaker=speaker,
                )
                output_dir = Path("outputs")
                output_dir.mkdir(parents=True, exist_ok=True)
                reference_audio_out = output_dir / "reference_demo.wav"
                lang = None if language == "Auto" else language
                result = pipeline.run(
                    original_text=text.strip(),
                    user_audio_path=user_audio_path,
                    reference_audio_out=str(reference_audio_out),
                    language=lang,
                    correction_prompt=correction_prompt,
                )
                corrected = "[提示] GPU 初始化失败，已自动回退 CPU。\n\n" + result.correction_text
                return result.reference_audio_path, result.correction_prompt, corrected
            except Exception as e2:
                return None, "", f"执行失败（GPU 回退 CPU 也失败）: {type(e2).__name__}: {e2}"
        return None, "", f"执行失败: {type(e).__name__}: {e}"


def create_demo() -> gr.Blocks:
    """Create Gradio interface."""
    theme = gr.themes.Soft()

    with gr.Blocks(theme=theme, title="Pronunciation Correction with OpenVINO") as demo:
        gr.Markdown(
            """
## 发音纠错助手 (OpenVINO + Qwen)

流程说明:
1. 输入目标文本，系统先用 Qwen3-TTS生成标准示范音频。
2. 上传或录制你的跟读音频。
3. 使用 Qwen-Audio 对比文本与音频，输出发音纠错建议。
"""
        )

        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(label="待练习文本", lines=5, placeholder="请输入你要练习朗读的句子")
                user_audio = gr.Audio(label="跟读音频", sources=["upload", "microphone"], type="filepath")
                correction_prompt = gr.Textbox(
                    label="纠错 Prompt",
                    lines=8,
                    value=build_default_prompt(""),
                )
                with gr.Row():
                    language = gr.Dropdown(
                        label="语言提示",
                        choices=["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"],
                        value="Auto",
                    )
                    speaker = gr.Dropdown(
                        label="示范音色",
                        choices=["vivian", "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu"],
                        value="vivian",
                    )

            with gr.Column(scale=1):
                qwen_audio_model = gr.Textbox(
                    label="Qwen-Audio 模型路径或模型 ID",
                    value="model/qwen2-audio-7b-ov-int4",
                )
                tts_model_dir = gr.Textbox(
                    label="TTS OpenVINO 模型目录",
                    value="model/Qwen3-TTS-CustomVoice-0.6B-fp16-ov",
                )
                device = gr.Dropdown(label="OpenVINO 设备", choices=["CPU"], value="CPU")
                with gr.Row():
                    gen_ref_btn = gr.Button("生成标准示范音频")
                    fill_prompt_btn = gr.Button("填充默认纠错 Prompt")
                run_btn = gr.Button("开始纠错", variant="primary")

        status_text = gr.Textbox(label="状态", lines=2)

        with gr.Row():
            reference_audio = gr.Audio(label="标准示范音频", type="filepath")

        correction_text = gr.Textbox(label="纠错结果", lines=12)

        text.change(
            fn=build_default_prompt,
            inputs=[text],
            outputs=[correction_prompt],
        )

        fill_prompt_btn.click(
            fn=build_default_prompt,
            inputs=[text],
            outputs=[correction_prompt],
        )

        gen_ref_btn.click(
            fn=generate_reference_audio_only,
            inputs=[text, tts_model_dir, device, language, speaker],
            outputs=[reference_audio, status_text],
        )

        run_btn.click(
            fn=run_pronunciation_correction,
            inputs=[text, user_audio, correction_prompt, qwen_audio_model, tts_model_dir, device, language, speaker],
            outputs=[reference_audio, correction_prompt, correction_text],
        )

    return demo


if __name__ == "__main__":
    app = create_demo()
    app.queue().launch(server_name="0.0.0.0", server_port=7860)
