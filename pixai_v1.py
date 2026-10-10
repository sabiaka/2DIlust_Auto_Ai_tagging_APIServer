"""PixAI v1.0 ONNX FP16 inference, using the existing Hugging Face cache."""

import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image

REPO = "Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8"
REVISION = "0800778563144a0e6fdf41ddadd84aae3cb0dbcf"
CATEGORIES = {0: "general", 1: "style", 3: "copyright", 4: "character", 5: "meta", 9: "rating"}
THRESHOLDS = {"general": .17, "style": .15, "copyright": .24, "character": .27, "meta": .17, "rating": .41}
session = None
tags = []


def load_model():
    global session, tags
    if session is not None:
        return

    # torch provides the CUDA/cuDNN DLLs on Windows.
    import torch
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = os.getenv("PIXAI_MODEL_PATH")
    if model_path:
        model_path = Path(model_path)
    else:
        model_path = Path(hf_hub_download(REPO, "model_fp16.onnx", revision=REVISION))
        hf_hub_download(REPO, "selected_tags.csv", revision=REVISION)
    with (model_path.parent / "selected_tags.csv").open(encoding="utf-8-sig", newline="") as f:
        loaded_tags = [(r["name"], CATEGORIES[int(r["category"])]) for r in csv.DictReader(f)]
    if len(loaded_tags) != 30877:
        raise ValueError("PixAI v1.0 requires exactly 30,877 tags.")

    use_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
    providers = [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    loaded_session = ort.InferenceSession(str(model_path), providers=providers)
    if use_gpu and "CUDAExecutionProvider" not in loaded_session.get_providers():
        raise RuntimeError("PixAI v1.0 CUDA initialization failed; check CUDA/cuDNN dependencies.")
    tags = loaded_tags
    session = loaded_session


def preprocess(image):
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    image = Image.alpha_composite(white, rgba).convert("RGB")
    scale = 1008 / max(image.size)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (1008, 1008), (0, 0, 0))
    canvas.paste(image, ((1008 - width) // 2, (1008 - height) // 2))
    array = (np.asarray(canvas, dtype=np.float32) / 255.0 - .5) / .5
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def predict_details(image, general_threshold, character_threshold):
    if session is None:
        raise RuntimeError("PixAI v1.0 model is not loaded.")
    thresholds = {
        category: float(os.getenv(f"PIXAI_{category.upper()}_THRESHOLD", default))
        for category, default in THRESHOLDS.items()
    }
    thresholds.update(general=general_threshold, character=character_threshold)
    probabilities = session.run(["prediction"], {"input": preprocess(image)})[0][0]
    if probabilities.shape != (len(tags),) or not np.isfinite(probabilities).all():
        raise RuntimeError("PixAI v1.0 returned invalid probabilities.")
    # Sigmoid is already included in this ONNX graph.
    result = [
        {"tag": name, "category": category, "score": float(probability)}
        for (name, category), probability in zip(tags, probabilities)
        if probability >= thresholds[category]
    ]
    priority = {"character": 0, "general": 1, "copyright": 2, "style": 3, "meta": 4, "rating": 5}
    result.sort(key=lambda item: (priority[item["category"]], -item["score"]))
    return result
