import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import numpy as np
from PIL import Image

logger = logging.getLogger("uvicorn")

MODEL_PATH = Path("./models_data")
SAFETENSORS_MODEL_FILE = MODEL_PATH / "model.safetensors"
ONNX_MODEL_FILE = MODEL_PATH / "model.onnx"
CONFIG_FILE = MODEL_PATH / "config.json"
TAG_FILE = MODEL_PATH / "selected_tags.csv"
EXTRA_TAG_TRANSLATION_FILE = Path(os.getenv("EXTRA_TAG_TRANSLATION_FILE", MODEL_PATH / "tag_translations_extra.csv"))
GENERATED_TRANSLATION_FILES = [
    MODEL_PATH / "pixai_missing_character_translations.csv",
    MODEL_PATH / "pixai_missing_translations.csv",
]

TAGGER_BACKEND = os.getenv("TAGGER_BACKEND", "pixai").strip().lower()
PIXAI_MODEL_NAME = os.getenv("PIXAI_MODEL_NAME", "v0.9")
PIXAI_GENERAL_THRESHOLD = float(os.getenv("PIXAI_GENERAL_THRESHOLD", "0.30"))
PIXAI_CHARACTER_THRESHOLD = float(os.getenv("PIXAI_CHARACTER_THRESHOLD", "0.75"))
WD_THRESHOLD = float(os.getenv("WD_THRESHOLD", "0.35"))

ort_session: Optional[object] = None
tags_list: List[str] = []
translation_map: Dict[str, str] = {}
input_size: int = 448
pixai_tagger = None


class TagDetail(TypedDict, total=False):
    tag: str
    prompt_tag: str
    translated: str
    category: str
    score: Optional[float]


def _normalize_tag(tag: str) -> str:
    return tag.replace("_", " ").strip()


def _translate_tag(tag: str) -> str:
    normalized = _normalize_tag(tag)
    return translation_map.get(normalized, normalized)


def _load_translation_map() -> None:
    tags_list.clear()
    translation_map.clear()

    if TAG_FILE.exists():
        _load_translation_csv(TAG_FILE, append_to_tags_list=True)
    else:
        logger.warning("%s is missing; built-in Japanese translations will be skipped.", TAG_FILE)

    if EXTRA_TAG_TRANSLATION_FILE.exists():
        _load_translation_csv(EXTRA_TAG_TRANSLATION_FILE, append_to_tags_list=False)
        logger.info("Loaded extra tag translations from %s.", EXTRA_TAG_TRANSLATION_FILE)

    for path in GENERATED_TRANSLATION_FILES:
        if path.exists():
            _load_translation_csv(path, append_to_tags_list=False)


def reload_translations() -> None:
    _load_translation_map()


def _load_translation_csv(path: Path, append_to_tags_list: bool) -> None:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        name_index = header.index("name")
        japanese_name_index = header.index("japanese_name") if "japanese_name" in header else -1

        for row in reader:
            try:
                english_tag = _normalize_tag(row[name_index])
                if append_to_tags_list:
                    tags_list.append(english_tag)
                if japanese_name_index != -1 and row[japanese_name_index]:
                    translation_map[english_tag] = row[japanese_name_index]
            except IndexError:
                continue


def ensure_onnx_model_exists() -> None:
    if ONNX_MODEL_FILE.exists():
        logger.info("Found %s; skipping ONNX export.", ONNX_MODEL_FILE)
        return

    if not SAFETENSORS_MODEL_FILE.exists():
        raise FileNotFoundError(f"{SAFETENSORS_MODEL_FILE} not found.")

    import json

    import timm
    import torch
    from safetensors.torch import load_file

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    logger.info("Exporting legacy WD tagger to ONNX...")
    model = timm.create_model(
        "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
        pretrained=False,
        num_classes=config["n_tags"],
    )
    state_dict = load_file(SAFETENSORS_MODEL_FILE, device="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    image_size = config.get("image_size", 448)
    dummy_input = torch.randn(1, 3, image_size, image_size, requires_grad=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_MODEL_FILE),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logger.info("Saved ONNX model to %s.", ONNX_MODEL_FILE)


def load_model_and_tags() -> None:
    global ort_session, input_size, pixai_tagger

    _load_translation_map()

    if TAGGER_BACKEND == "pixai":
        try:
            from imgutils.tagging.pixai import get_pixai_tags
        except ImportError as exc:
            raise RuntimeError(
                "TAGGER_BACKEND=pixai requires dghs-imgutils. "
                "Install requirements.txt or set TAGGER_BACKEND=wd to use the legacy model."
            ) from exc

        pixai_tagger = get_pixai_tags
        logger.info(
            "TAGGER_BACKEND=pixai: using PixAI Tagger %s (general threshold %.2f, character threshold %.2f).",
            PIXAI_MODEL_NAME,
            PIXAI_GENERAL_THRESHOLD,
            PIXAI_CHARACTER_THRESHOLD,
        )
        return

    if TAGGER_BACKEND != "wd":
        raise ValueError("TAGGER_BACKEND must be either 'pixai' or 'wd'.")

    import onnxruntime

    ensure_onnx_model_exists()
    device = "cuda" if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else "cpu"
    providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    logger.info("Loading legacy WD ONNX model from %s on %s...", ONNX_MODEL_FILE, device)
    ort_session = onnxruntime.InferenceSession(str(ONNX_MODEL_FILE), providers=providers)
    input_size = ort_session.get_inputs()[0].shape[-1]


def prepare_image(image: Image.Image, target_size: int) -> np.ndarray:
    import torchvision.transforms.functional as TVF

    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2
    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
    image_tensor = TVF.to_tensor(padded_image).unsqueeze(0)
    image_tensor = TVF.normalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return image_tensor.numpy()


def _predict_pixai(image: Image.Image) -> List[str]:
    if pixai_tagger is None:
        raise RuntimeError("PixAI tagger is not loaded. Call load_model_and_tags() first.")

    thresholds = {
        "general": PIXAI_GENERAL_THRESHOLD,
        "character": PIXAI_CHARACTER_THRESHOLD,
    }
    general_tags, character_tags, ips = pixai_tagger(
        image.convert("RGB"),
        model_name=PIXAI_MODEL_NAME,
        thresholds=thresholds,
        fmt=("general", "character", "ips"),
    )

    ordered_tags: List[str] = []
    for tag_group in (character_tags, general_tags):
        ordered_tags.extend(tag_group.keys())
    ordered_tags.extend(ips)

    seen = set()
    translated_tags = []
    for tag in ordered_tags:
        translated = _translate_tag(tag)
        if translated not in seen:
            seen.add(translated)
            translated_tags.append(translated)
    return translated_tags


def _predict_pixai_details(image: Image.Image) -> List[TagDetail]:
    if pixai_tagger is None:
        raise RuntimeError("PixAI tagger is not loaded. Call load_model_and_tags() first.")

    thresholds = {
        "general": PIXAI_GENERAL_THRESHOLD,
        "character": PIXAI_CHARACTER_THRESHOLD,
    }
    general_tags, character_tags, ips = pixai_tagger(
        image.convert("RGB"),
        model_name=PIXAI_MODEL_NAME,
        thresholds=thresholds,
        fmt=("general", "character", "ips"),
    )

    details: List[TagDetail] = []
    seen = set()

    def append_detail(tag: str, category: str, score: Optional[float]) -> None:
        normalized = _normalize_tag(tag)
        if normalized in seen:
            return
        seen.add(normalized)
        details.append(
            {
                "tag": normalized,
                "prompt_tag": normalized.replace(" ", "_"),
                "translated": _translate_tag(normalized),
                "category": category,
                "score": float(score) if score is not None else None,
            }
        )

    for tag, score in character_tags.items():
        append_detail(tag, "character", score)
    for tag, score in general_tags.items():
        append_detail(tag, "general", score)
    for tag in ips:
        append_detail(tag, "copyright", None)

    return details


def _predict_wd(image: Image.Image) -> List[str]:
    if ort_session is None:
        raise RuntimeError("Legacy WD model is not loaded. Call load_model_and_tags() first.")

    image_np = prepare_image(image.convert("RGB"), input_size)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    ort_outs = ort_session.run([output_name], {input_name: image_np})[0]
    preds = 1 / (1 + np.exp(-ort_outs))
    tag_preds = preds[0]
    predicted_tags = [tag for tag, score in zip(tags_list, tag_preds) if score > WD_THRESHOLD]
    return [_translate_tag(tag) for tag in predicted_tags]


def _predict_wd_details(image: Image.Image) -> List[TagDetail]:
    if ort_session is None:
        raise RuntimeError("Legacy WD model is not loaded. Call load_model_and_tags() first.")

    image_np = prepare_image(image.convert("RGB"), input_size)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    ort_outs = ort_session.run([output_name], {input_name: image_np})[0]
    preds = 1 / (1 + np.exp(-ort_outs))
    tag_preds = preds[0]
    return [
        {
            "tag": tag,
            "prompt_tag": tag.replace(" ", "_"),
            "translated": _translate_tag(tag),
            "category": "general",
            "score": float(score),
        }
        for tag, score in zip(tags_list, tag_preds)
        if score > WD_THRESHOLD
    ]


def predict(image: Image.Image) -> List[str]:
    if TAGGER_BACKEND == "pixai":
        return _predict_pixai(image)
    return _predict_wd(image)


def predict_details(image: Image.Image) -> List[TagDetail]:
    if TAGGER_BACKEND == "pixai":
        return _predict_pixai_details(image)
    return _predict_wd_details(image)


def get_tagger_info() -> Dict[str, object]:
    return {
        "backend": TAGGER_BACKEND,
        "pixai_model_name": PIXAI_MODEL_NAME if TAGGER_BACKEND == "pixai" else None,
        "pixai_general_threshold": PIXAI_GENERAL_THRESHOLD if TAGGER_BACKEND == "pixai" else None,
        "pixai_character_threshold": PIXAI_CHARACTER_THRESHOLD if TAGGER_BACKEND == "pixai" else None,
        "wd_threshold": WD_THRESHOLD if TAGGER_BACKEND == "wd" else None,
        "legacy_onnx_loaded": ort_session is not None,
        "pixai_loaded": pixai_tagger is not None,
    }
