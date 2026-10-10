"""Start the Python API from this checkout, without Docker."""
import argparse
import os
from pathlib import Path

def configuration():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVER_PORT", "8000")))
    parser.add_argument("--check", action="store_true", help="Print configuration without loading models")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    return args

def main():
    args = configuration()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    os.environ.setdefault("TAGGER_BACKEND", "pixai")
    os.environ.setdefault("PIXAI_MODEL_NAME", "v1.0")
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(root / ".cache" / "torch"))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if args.check:
        print(f"host={args.host} port={args.port} model={os.environ['PIXAI_MODEL_NAME']}")
        return
    # Load PyTorch before ONNX Runtime to make bundled CUDA DLLs available.
    import torch
    import onnxruntime
    if os.getenv("ORT_PRELOAD_NVIDIA") == "1":
        onnxruntime.preload_dlls(directory="")
    import uvicorn
    print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}", flush=True)
    uvicorn.run("main:app", host=args.host, port=args.port, workers=1)

if __name__ == "__main__":
    main()

