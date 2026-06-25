import os

from huggingface_hub import snapshot_download
from transformers import AutoProcessor


def main() -> None:
    model_name = os.getenv("QWEN_VL_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")
    cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE")
    print(f"Downloading {model_name} into Hugging Face cache...")
    snapshot_download(repo_id=model_name, cache_dir=cache_dir)
    AutoProcessor.from_pretrained(model_name)
    print("Download complete.")


if __name__ == "__main__":
    main()
