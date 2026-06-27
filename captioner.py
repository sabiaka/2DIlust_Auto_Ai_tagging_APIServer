import json
import logging
import os
import re
import threading
from typing import Dict, List, TypedDict

from PIL import Image

logger = logging.getLogger("uvicorn")

QWEN_VL_MODEL_NAME = os.getenv("QWEN_VL_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
QWEN_VL_MAX_NEW_TOKENS = int(os.getenv("QWEN_VL_MAX_NEW_TOKENS", "384"))
QWEN_VL_TEMPERATURE = float(os.getenv("QWEN_VL_TEMPERATURE", "0.15"))
QWEN_VL_DEVICE_MAP = os.getenv("QWEN_VL_DEVICE_MAP", "auto").strip()
QWEN_VL_LOAD_IN_4BIT = os.getenv("QWEN_VL_LOAD_IN_4BIT", "1").strip().lower() not in {"0", "false", "no"}
QWEN_VL_MAX_PIXELS = int(os.getenv("QWEN_VL_MAX_PIXELS", str(768 * 28 * 28)))
QWEN_VL_MIN_PIXELS = int(os.getenv("QWEN_VL_MIN_PIXELS", str(128 * 28 * 28)))
QWEN_VL_MAX_IMAGE_SIDE = int(os.getenv("QWEN_VL_MAX_IMAGE_SIDE", "1024"))

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


def _extract_first_json_object(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return text[start:]


def _parse_description(text: str) -> ImageDescription:
    try:
        data = json.loads(_extract_first_json_object(text))
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


def _description_prompt(tag_hints: List[str]) -> str:
    instruction = (
        "\u5fc5\u305a\u65e5\u672c\u8a9e\u3060\u3051\u3067\u56de\u7b54\u3057\u3066\u304f\u3060\u3055\u3044\u3002\u82f1\u8a9e\u306f\u7981\u6b62\u3067\u3059\u3002"
        "\u753b\u50cf\u3092\u76f4\u63a5\u89b3\u5bdf\u3057\u3001\u30bf\u30b0\u306e\u7f85\u5217\u3067\u306f\u306a\u304f\u81ea\u7136\u306a\u6587\u7ae0\u3067\u8aac\u660e\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
        "\u8868\u60c5\u3001\u8996\u7dda\u3001\u53e3\u5143\u3001\u7709\u3001\u59ff\u52e2\u3001\u5834\u9762\u3001\u69cb\u56f3\u3001\u7a7a\u6c17\u611f\u3092\u3001\u898b\u3048\u3066\u3044\u308b\u7bc4\u56f2\u3060\u3051\u306b\u57fa\u3065\u3044\u3066\u5177\u4f53\u7684\u306b\u66f8\u3044\u3066\u304f\u3060\u3055\u3044\u3002"
        "\u8fd4\u7b54\u306fJSON\u30aa\u30d6\u30b8\u30a7\u30af\u30c81\u3064\u3060\u3051\u306b\u3057\u3066\u304f\u3060\u3055\u3044\u3002\u30ad\u30fc\u306f description, expression, situation \u306e3\u3064\u3060\u3051\u3067\u3059\u3002"
        "\u5404\u5024\u306b\u306f\u3001\u3053\u306e\u753b\u50cf\u305d\u306e\u3082\u306e\u306b\u3064\u3044\u3066\u306e\u5177\u4f53\u7684\u306a\u65e5\u672c\u8a9e\u6587\u3092\u5165\u308c\u3066\u304f\u3060\u3055\u3044\u3002"
        "\u8aac\u660e\u7528\u306e\u30d7\u30ec\u30fc\u30b9\u30db\u30eb\u30c0\u30fc\u3084\u30b9\u30ad\u30fc\u30de\u8aac\u660e\u3092\u5024\u306b\u5165\u308c\u3066\u306f\u3044\u3051\u307e\u305b\u3093\u3002"
    )
    hint_text = ", ".join(tag_hints[:80])
    if hint_text:
        instruction += (
            "\n\u53c2\u8003\u30bf\u30b0\uff08\u88dc\u52a9\u60c5\u5831\u3067\u3059\u3002\u753b\u50cf\u306e\u76f4\u63a5\u89b3\u5bdf\u3092\u512a\u5148\u3057\u3066\u304f\u3060\u3055\u3044\uff09: "
            f"{hint_text}"
        )
    return instruction


def describe_image(image: Image.Image, tag_hints: List[str]) -> ImageDescription:
    _load_captioner()

    if _caption_model is None or _caption_processor is None:
        raise RuntimeError("Qwen2.5-VL captioner failed to load.")

    rgb_image = _resize_for_captioner(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": rgb_image},
                {"type": "text", "text": _description_prompt(tag_hints)},
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
