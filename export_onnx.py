# export_onnx.py
import torch
import timm
from safetensors.torch import load_file
import json
from pathlib import Path
import logging

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- グローバル変数 ---
MODEL_PATH = Path('./models_data')
SAFETENSORS_MODEL_FILE = MODEL_PATH / 'model.safetensors'
CONFIG_FILE = MODEL_PATH / 'config.json'
ONNX_MODEL_FILE = MODEL_PATH / 'model.onnx'

def export_to_onnx():
    """
    safetensorsモデルをONNX形式に変換するスクリプト
    """
    if not SAFETENSORS_MODEL_FILE.exists():
        logging.error(f"❌ モデルファイル `{SAFETENSORS_MODEL_FILE}` が見つかりません。")
        return
    if not CONFIG_FILE.exists():
        logging.error(f"❌ 設定ファイル `{CONFIG_FILE}` が見つかりません。")
        return

    logging.info("モデルのロードちう... 🚀")
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    # timmでモデルの骨格を作成
    model = timm.create_model(
        'eva02_large_patch14_448.mim_in22k_ft_in22k_in1k',
        pretrained=False,
        num_classes=config['n_tags']
    )

    # 重みをロード
    state_dict = load_file(SAFETENSORS_MODEL_FILE, device='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    logging.info("モデルのロード完了！✨")

    # ONNXに変換するためのダミー入力データを作成
    image_size = config.get('image_size', 448)
    dummy_input = torch.randn(1, 3, image_size, image_size, requires_grad=True)

    logging.info("ONNXへの変換開始！ちょっと待っててね...")
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_MODEL_FILE),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input' : {0 : 'batch_size'},
                      'output' : {0 : 'batch_size'}}
    )
    logging.info(f"🎉 やったー！`{ONNX_MODEL_FILE}` にONNXモデルを保存したよ！ 🎉")

if __name__ == '__main__':
    export_to_onnx()