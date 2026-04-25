"""
Pronunciation correction pipeline for OpenVINO workshop baseline.

Pipeline steps:
1) Generate reference pronunciation audio from input text (Baseline OVQwen3TTSModel).
2) Build a multimodal correction prompt with original text and scoring instructions.
3) Run Qwen-Audio correction with OpenVINO backend (optimum-intel OVModelForCausalLM).
4) Return correction feedback text.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write
from transformers import AutoModelForCausalLM, AutoProcessor

from optimum.intel.openvino import OVModelForCausalLM


def _ensure_lab3_helper_importable(project_root: Path) -> None:
    """Ensure baseline TTS helper folder is importable."""
    lab3_dir = project_root / "lab3-text-to-speech"
    if str(lab3_dir) not in sys.path:
        sys.path.insert(0, str(lab3_dir))


def _ensure_qwen2_audio_helper_importable(project_root: Path) -> None:
    """Ensure qwen2-audio helper folder is importable."""
    qwen2_dir = project_root / "qwen2-audio"
    if str(qwen2_dir) not in sys.path:
        sys.path.insert(0, str(qwen2_dir))


def _is_qwen2_audio_ov_directory(model_path: str) -> bool:
    """Detect OpenVINO Qwen2-Audio split-IR directory layout."""
    p = Path(model_path)
    if not p.exists() or not p.is_dir():
        return False
    required = [
        "openvino_audio_embedding.xml",
        "openvino_mulimodal_projection_model.xml",
        "openvino_text_embedding_model.xml",
        "openvino_language_model.xml",
    ]
    return all((p / name).exists() for name in required)


def _normalize_float_audio(audio: np.ndarray) -> np.ndarray:
    """Convert input waveform to mono float32 in [-1, 1]."""
    x = np.asarray(audio)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        scale = max(abs(info.min), info.max)
        x = x.astype(np.float32) / float(scale)
    else:
        x = x.astype(np.float32)
    return np.clip(x, -1.0, 1.0)


def _load_audio_mono_resampled(audio_path: str, target_sr: int) -> np.ndarray:
    """Load audio as mono float32 and resample to target sampling rate."""
    try:
        import soundfile as sf

        audio, sr = sf.read(audio_path)
        audio = _normalize_float_audio(audio)
        if int(sr) != int(target_sr):
            import librosa

            audio = librosa.resample(audio, orig_sr=int(sr), target_sr=int(target_sr))
        return audio.astype(np.float32)
    except Exception:
        # Fast path for wav files without relying on external backends (e.g. sox).
        try:
            sr, audio = wav_read(audio_path)
            audio = _normalize_float_audio(audio)
            if int(sr) != int(target_sr):
                import librosa

                audio = librosa.resample(audio, orig_sr=int(sr), target_sr=int(target_sr))
            return audio.astype(np.float32)
        except Exception:
            pass

        import librosa

        audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio.astype(np.float32)


def _prepare_audio_processor_inputs(processor, text_for_model: str, audio_np: np.ndarray, sampling_rate: int):
    """Build processor inputs with robust compatibility for different processor signatures."""
    base_kwargs = {
        "text": text_for_model,
        "return_tensors": "pt",
        "padding": True,
    }
    # Different processor versions use different keyword names and accepted shapes.
    candidates = [
        {"audios": [audio_np], "sampling_rate": sampling_rate},
        {"audio": [audio_np], "sampling_rate": sampling_rate},
        {"audio": audio_np, "sampling_rate": sampling_rate},
        {"audios": [audio_np]},
        {"audio": [audio_np]},
        {"audio": audio_np},
    ]

    last_error = None
    for audio_kwargs in candidates:
        try:
            inputs = processor(**base_kwargs, **audio_kwargs)
            keys = set(inputs.keys()) if hasattr(inputs, "keys") else set()
            # Qwen2-Audio processors typically emit input_features / feature_attention_mask.
            if "input_features" in keys or "feature_attention_mask" in keys:
                return inputs
            # Fallback for other multimodal processors that may expose different names.
            if any(("audio" in k) or ("feature" in k) for k in keys):
                return inputs
            last_error = RuntimeError(f"Processor call succeeded but no audio features found in keys: {sorted(keys)}")
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to prepare multimodal audio inputs for processor: {last_error}")


def _to_int16(audio_float: np.ndarray) -> np.ndarray:
    """Convert float waveform in [-1, 1] to int16."""
    clipped = np.clip(audio_float, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


@dataclass
class PronunciationResult:
    reference_audio_path: str
    correction_prompt: str
    correction_text: str


class QwenAudioOVCorrector:
    """Qwen-Audio pronunciation corrector with OpenVINO inference."""

    def __init__(
        self,
        model_id_or_path: str,
        device: str = "CPU",
        max_new_tokens: int = 512,
        export_if_needed: bool = False,
        project_root: Optional[str] = None,
    ):
        self.model_id_or_path = model_id_or_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.backend = "optimum_ov_causallm"

        self.processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)

        if _is_qwen2_audio_ov_directory(model_id_or_path):
            if project_root is None:
                project_root = str(Path(__file__).resolve().parent)
            _ensure_qwen2_audio_helper_importable(Path(project_root))
            OVQwen2AudioForConditionalGeneration = importlib.import_module("ov_qwen2_audio_helper").OVQwen2AudioForConditionalGeneration

            # Newer transformers expects this class-level flag for generate() path.
            if not hasattr(OVQwen2AudioForConditionalGeneration, "_is_stateful"):
                OVQwen2AudioForConditionalGeneration._is_stateful = False

            self.model = OVQwen2AudioForConditionalGeneration(Path(model_id_or_path), device)
            self.backend = "ov_qwen2_audio_helper"
        else:
            self.model = OVModelForCausalLM.from_pretrained(
                model_id_or_path,
                device=device,
                trust_remote_code=True,
                export=export_if_needed,
            )

        feature_extractor = getattr(self.processor, "feature_extractor", None)
        self.sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))

    @staticmethod
    def build_correction_prompt(original_text: str) -> str:
        """Build correction prompt with fixed rubric."""
        return (
        "你是一位严格且专业的发音教练。请听用户的录音，并将其与下方的“目标文本”进行对比，提供详细的纠错反馈。\n"
        "【重要指令】：请全程使用中文（简体）进行回答，并严格按照下方的【评分基准】进行客观打分。\n\n"
        "【评分基准】（采用满分100分的扣分制）：\n"
        "- 基础分为 100 分。\n"
        "- 严重发音错误：每一个完全读错或导致意思改变的单词，扣 5 分。\n"
        "- 轻微发音瑕疵：每一个元音/辅音发音不到位但不影响理解的单词，扣 2 分。\n"
        "- 漏读或多读：每一个漏掉或多加的单词，扣 3 分。\n"
        "- 语调与节奏：若整体重音放错、结巴或断句极其不自然，根据严重程度整体扣 5 到 10 分。\n"
        "-（注：最低得分为0分，不设负分）\n\n"
        "目标文本：\n"
        f"{original_text}\n\n"
        "请严格按照以下格式输出你的反馈：\n"
        "1. 综合评分：[填写最终分数]分。（请简要列出计算公式，例如：100 - 严重错误10分 - 漏读3分 = 87分）。\n"
        "2. 发音错误：明确指出读错的字、词，并给出正确的发音建议。\n"
        "3. 语调与节奏问题：指出重音错误、停顿不自然等现象。\n"
        "4. 漏读或多读的内容：列出具体的单词。\n"
        "5. 改进建议：一小段简短的练习指导。"
)

    def infer_correction(
        self,
        original_text: str,
        user_audio_path: str,
        correction_prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        """Run multimodal inference with prompt + user audio."""
        prompt = correction_prompt.strip() if correction_prompt and correction_prompt.strip() else self.build_correction_prompt(original_text)
        audio_np = _load_audio_mono_resampled(user_audio_path, target_sr=self.sampling_rate)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio_url": user_audio_path},
                ],
            }
        ]

        text_for_model = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = _prepare_audio_processor_inputs(
            processor=self.processor,
            text_for_model=text_for_model,
            audio_np=audio_np,
            sampling_rate=self.sampling_rate,
        )

        generation_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
        )

        prompt_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        new_tokens = generation_ids[:, prompt_len:]
        correction_text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        return {
            "prompt": prompt,
            "correction_text": correction_text,
        }


class QwenAudioCUDACorrector:
    """Qwen-Audio pronunciation corrector with PyTorch CUDA inference."""

    def __init__(
        self,
        model_id_or_path: str,
        device: str = "cuda:0",
        max_new_tokens: int = 512,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please install CUDA-enabled PyTorch and NVIDIA driver.")

        self.model_id_or_path = model_id_or_path
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)

        dtype = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()

        feature_extractor = getattr(self.processor, "feature_extractor", None)
        self.sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))

    def infer_correction(
        self,
        original_text: str,
        user_audio_path: str,
        correction_prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        """Run multimodal inference with prompt + user audio on CUDA."""
        prompt = correction_prompt.strip() if correction_prompt and correction_prompt.strip() else QwenAudioOVCorrector.build_correction_prompt(original_text)
        audio_np = _load_audio_mono_resampled(user_audio_path, target_sr=self.sampling_rate)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio_url": user_audio_path},
                ],
            }
        ]

        text_for_model = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = _prepare_audio_processor_inputs(
            processor=self.processor,
            text_for_model=text_for_model,
            audio_np=audio_np,
            sampling_rate=self.sampling_rate,
        )

        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generation_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )

        prompt_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        new_tokens = generation_ids[:, prompt_len:]
        correction_text = self.processor.batch_decode(new_tokens.detach().cpu(), skip_special_tokens=True)[0].strip()

        return {
            "prompt": prompt,
            "correction_text": correction_text,
        }


class PronunciationCorrectionPipeline:
    """End-to-end pronunciation correction pipeline."""

    def __init__(
        self,
        project_root: str,
        tts_model_dir: str,
        qwen_audio_model_dir: Optional[str] = None,
        device: str = "CPU",
        tts_speaker: str = "vivian",
    ):
        self.project_root = Path(project_root).resolve()
        self.tts_speaker = tts_speaker
        self.qwen_audio_model_dir = qwen_audio_model_dir
        self.device = device.strip()
        normalized_device = self.device.upper()
        self.is_nvidia_mode = normalized_device in {"NVIDIA", "CUDA", "CUDA:0"}

        # Baseline TTS wrapper is OpenVINO-based; on NVIDIA mode we keep TTS on CPU.
        self.tts_device = "CPU" if self.is_nvidia_mode else self.device

        _ensure_lab3_helper_importable(self.project_root)
        OVQwen3TTSModel = importlib.import_module("qwen_3_tts_helper").OVQwen3TTSModel

        self.tts_model = OVQwen3TTSModel.from_pretrained(
            model_dir=tts_model_dir,
            device=self.tts_device,
        )
        self.corrector: Optional[Any] = None

        if qwen_audio_model_dir:
            self.corrector = self._build_corrector()

    def _build_corrector(self):
        """Build Qwen-Audio corrector backend based on selected device."""
        if not self.qwen_audio_model_dir:
            raise ValueError("Qwen-Audio model path/id is required for correction inference.")

        if self.is_nvidia_mode:
            return QwenAudioCUDACorrector(
                model_id_or_path=self.qwen_audio_model_dir,
                device="cuda:0",
                max_new_tokens=512,
            )

        return QwenAudioOVCorrector(
            model_id_or_path=self.qwen_audio_model_dir,
            device=self.device,
            max_new_tokens=512,
            export_if_needed=False,
            project_root=str(self.project_root),
        )

    def _ensure_corrector(self) -> Any:
        """Lazy-load Qwen-Audio corrector when needed."""
        if self.corrector is None:
            self.corrector = self._build_corrector()
        return self.corrector

    def generate_reference_audio(
        self,
        text: str,
        output_wav_path: str,
        language: Optional[str] = None,
        instruct: Optional[str] = "Read clearly with standard pronunciation.",
    ) -> str:
        """Generate standard pronunciation audio using baseline OpenVINO TTS wrapper."""
        wavs, sr = self.tts_model.generate_custom_voice(
            text=text,
            language=language,
            speaker=self.tts_speaker,
            instruct=instruct,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )

        out_path = Path(output_wav_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wav_write(str(out_path), int(sr), _to_int16(np.asarray(wavs[0], dtype=np.float32)))
        return str(out_path)

    def run(
        self,
        original_text: str,
        user_audio_path: str,
        reference_audio_out: str,
        language: Optional[str] = None,
        correction_prompt: Optional[str] = None,
    ) -> PronunciationResult:
        """Execute full pipeline and return correction result."""
        if not Path(user_audio_path).exists():
            raise FileNotFoundError(f"User audio not found: {user_audio_path}")

        ref_audio = self.generate_reference_audio(
            text=original_text,
            output_wav_path=reference_audio_out,
            language=language,
        )
        corrector = self._ensure_corrector()
        correction = corrector.infer_correction(
            original_text=original_text,
            user_audio_path=user_audio_path,
            correction_prompt=correction_prompt,
        )

        return PronunciationResult(
            reference_audio_path=ref_audio,
            correction_prompt=correction["prompt"],
            correction_text=correction["correction_text"],
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenVINO pronunciation correction pipeline")
    parser.add_argument("--text", type=str, required=True, help="Target practice text")
    parser.add_argument("--user-audio", type=str, required=True, help="Path to learner audio file")
    parser.add_argument(
        "--qwen-audio-model",
        type=str,
        required=True,
        help="Qwen-Audio model id or local OpenVINO model directory",
    )
    parser.add_argument(
        "--tts-model-dir",
        type=str,
        default="lab3-text-to-speech/Qwen3-TTS-CustomVoice-0.6B-fp16-ov",
        help="Baseline Qwen3-TTS OpenVINO model directory",
    )
    parser.add_argument("--device", type=str, default="CPU", help="Runtime device: CPU/GPU/NPU (OpenVINO)")
    parser.add_argument("--language", type=str, default=None, help="Language hint for TTS")
    parser.add_argument(
        "--speaker",
        type=str,
        default="vivian",
        help="Speaker for baseline custom_voice model",
    )
    parser.add_argument(
        "--reference-audio-out",
        type=str,
        default="outputs/reference_demo.wav",
        help="Output path for generated reference audio",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parent

    pipeline = PronunciationCorrectionPipeline(
        project_root=str(project_root),
        tts_model_dir=args.tts_model_dir,
        qwen_audio_model_dir=args.qwen_audio_model,
        device=args.device,
        tts_speaker=args.speaker,
    )

    result = pipeline.run(
        original_text=args.text,
        user_audio_path=args.user_audio,
        reference_audio_out=args.reference_audio_out,
        language=args.language,
    )

    print("=" * 80)
    print("Reference audio generated:")
    print(result.reference_audio_path)
    print("=" * 80)
    print("Correction prompt:")
    print(result.correction_prompt)
    print("=" * 80)
    print("Qwen-Audio correction result:")
    print(result.correction_text)
    print("=" * 80)


if __name__ == "__main__":
    main()
