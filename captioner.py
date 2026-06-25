import json
import logging
import os
import re
import threading
from typing import Dict, List, Optional, TypedDict

from PIL import Image

logger = logging.getLogger("uvicorn")

QWEN_VL_MODEL_NAME = os.getenv("QWEN_VL_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
QWEN_VL_MAX_NEW_TOKENS = int(os.getenv("QWEN_VL_MAX_NEW_TOKENS", "256"))
QWEN_VL_TEMPERATURE = float(os.getenv("QWEN_VL_TEMPERATURE", "0.2"))
QWEN_VL_DEVICE_MAP = os.getenv("QWEN_VL_DEVICE_MAP", "auto").strip()
QWEN_VL_LOAD_IN_4BIT = os.getenv("QWEN_VL_LOAD_IN_4BIT", "1").strip().lower() not in {"0", "false", "no"}
QWEN_VL_MAX_PIXELS = int(os.getenv("QWEN_VL_MAX_PIXELS", str(512 * 28 * 28)))
QWEN_VL_MIN_PIXELS = int(os.getenv("QWEN_VL_MIN_PIXELS", str(128 * 28 * 28)))
QWEN_VL_MAX_IMAGE_SIDE = int(os.getenv("QWEN_VL_MAX_IMAGE_SIDE", "768"))

_caption_lock = threading.Lock()
_caption_model = None
_caption_processor = None


class CaptionerOutOfMemoryError(RuntimeError):
    pass


class ImageDescription(TypedDict):
    description: str
    expression: str
    situation: str


def _load_captioner() -> None:
    global _caption_model, _caption_processor

    if _caption_model is not None and _caption_processor is not None:
        return

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    logger.info("Loading Qwen2.5-VL captioner: %s", QWEN_VL_MODEL_NAME)
    load_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if QWEN_VL_LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if QWEN_VL_DEVICE_MAP:
        load_kwargs["device_map"] = QWEN_VL_DEVICE_MAP

    _caption_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_VL_MODEL_NAME,
        **load_kwargs,
    )
    if not QWEN_VL_DEVICE_MAP and torch.cuda.is_available():
        _caption_model = _caption_model.to("cuda")
    _caption_processor = AutoProcessor.from_pretrained(
        QWEN_VL_MODEL_NAME,
        min_pixels=QWEN_VL_MIN_PIXELS,
        max_pixels=QWEN_VL_MAX_PIXELS,
    )
    logger.info("Qwen2.5-VL captioner loaded.")


def _device_for_inputs():
    import torch

    if _caption_model is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(_caption_model, "device"):
        return _caption_model.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _clean_json_text(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text


def _parse_description(text: str) -> ImageDescription:
    try:
        data = json.loads(_clean_json_text(text))
    except json.JSONDecodeError:
        return {
            "description": text.strip(),
            "expression": "",
            "situation": "",
        }

    return {
        "description": str(data.get("description") or "").strip(),
        "expression": str(data.get("expression") or "").strip(),
        "situation": str(data.get("situation") or "").strip(),
    }


def _resize_for_captioner(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest <= QWEN_VL_MAX_IMAGE_SIDE:
        return image

    scale = QWEN_VL_MAX_IMAGE_SIDE / longest
    next_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(next_size, Image.Resampling.LANCZOS)


def reset_captioner() -> None:
    global _caption_model, _caption_processor

    _caption_model = None
    _caption_processor = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.exception("Failed to clear CUDA cache after captioner reset.")


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
    except Exception:
        return "out of memory" in str(exc).lower()

    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def describe_image(image: Image.Image, tag_hints: List[str]) -> ImageDescription:
    _load_captioner()

    if _caption_model is None or _caption_processor is None:
        raise RuntimeError("Qwen2.5-VL captioner failed to load.")

    hint_text = ", ".join(tag_hints[:80])
    instruction = (
        "必ず日本語だけで回答してください。英語は禁止です。"
        "画像を直接観察し、タグの羅列ではなく自然な文章で説明してください。"
        "表情、視線、口元、眉、姿勢、場面、構図、空気感を、見えている範囲だけに基づいて具体的に書いてください。"
        "返答はJSONオブジェクト1つだけにしてください。キーは description, expression, situation の3つだけです。"
        "各値には、この画像そのものについての具体的な日本語文を入れてください。"
        "説明用のプレースホルダーやスキーマ説明を値に入れてはいけません。"
    )
    if hint_text:
        instruction += f"\n参考タグ（補助情報です。画像の直接観察を優先してください）: {hint_text}"

    rgb_image = _resize_for_captioner(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": rgb_image},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    with _caption_lock:
        text = _caption_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = _caption_processor(
            text=[text],
            images=[rgb_image],
            padding=True,
            return_tensors="pt",
        ).to(_device_for_inputs())
        generate_kwargs = {
            "max_new_tokens": QWEN_VL_MAX_NEW_TOKENS,
            "do_sample": QWEN_VL_TEMPERATURE > 0,
        }
        if QWEN_VL_TEMPERATURE > 0:
            generate_kwargs["temperature"] = QWEN_VL_TEMPERATURE

        try:
            generated_ids = _caption_model.generate(**inputs, **generate_kwargs)
        except RuntimeError as exc:
            if _is_cuda_oom(exc):
                reset_captioner()
                raise CaptionerOutOfMemoryError(
                    "Qwen2.5-VL ran out of GPU memory. Try lower QWEN_VL_MAX_PIXELS, "
                    "lower QWEN_VL_MAX_IMAGE_SIDE, or use a smaller/quantized model."
                ) from exc
            raise
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = _caption_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    return _parse_description(output_text)


def get_captioner_info() -> Dict[str, object]:
    return {
        "model_name": QWEN_VL_MODEL_NAME,
        "loaded": _caption_model is not None,
        "max_new_tokens": QWEN_VL_MAX_NEW_TOKENS,
        "temperature": QWEN_VL_TEMPERATURE,
        "device_map": QWEN_VL_DEVICE_MAP,
        "load_in_4bit": QWEN_VL_LOAD_IN_4BIT,
        "min_pixels": QWEN_VL_MIN_PIXELS,
        "max_pixels": QWEN_VL_MAX_PIXELS,
        "max_image_side": QWEN_VL_MAX_IMAGE_SIDE,
    }
