import torch
import timm
import onnxruntime
import numpy as np
import torchvision.transforms.functional as TVF
from PIL import Image
from pathlib import Path
from typing import List
import csv
import json
import logging

# ロガーの設定
logger = logging.getLogger("uvicorn")

# --- グローバル変数 ---
MODEL_PATH = Path('./models_data')
SAFETENSORS_MODEL_FILE = MODEL_PATH / 'model.safetensors'
ONNX_MODEL_FILE = MODEL_PATH / 'model.onnx'
CONFIG_FILE = MODEL_PATH / 'config.json'
TAG_FILE = MODEL_PATH / 'selected_tags.csv'

THRESHOLD = 0.35
# ONNX RuntimeでCUDAが使えるかチェック
DEVICE = 'cuda' if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else 'cpu'

# --- ONNXモデルがなければ自動で変換する関数 ---
def ensure_onnx_model_exists():
    """
    ONNXモデルがなかったら、safetensorsから自動で作っちゃう魔法の関数
    """
    if ONNX_MODEL_FILE.exists():
        logger.info(f"✅ `{ONNX_MODEL_FILE}` 発見！変換はスキップするね。")
        return

    logger.warning(f"⚠️ `{ONNX_MODEL_FILE}` が見つからない！今からsafetensorsから変換するよ。ちょっと待っててね…")

    if not SAFETENSORS_MODEL_FILE.exists():
        logger.error(f"❌ うそ…肝心の `{SAFETENSORS_MODEL_FILE}` もないじゃん！モデルをダウンロードしてきて！")
        raise FileNotFoundError(f"{SAFETENSORS_MODEL_FILE} not found.")
        
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ `{CONFIG_FILE}` がないよ！モデルと一緒に配布されてるはずだから確認して！")
        raise
        
    logger.info("モデルの骨格をtimmで作成中...")
    model = timm.create_model(
        'eva02_large_patch14_448.mim_in22k_ft_in22k_in1k',
        pretrained=False,
        num_classes=config['n_tags']
    )

    logger.info("safetensorsから魂（おもみ）をロード中...")
    from safetensors.torch import load_file
    state_dict = load_file(SAFETENSORS_MODEL_FILE, device='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    image_size = config.get('image_size', 448)
    dummy_input = torch.randn(1, 3, image_size, image_size, requires_grad=True)

    logger.info("✨ ONNXに変身開始！これが結構時間かかるかも… ✨")
    
    # ONNXへエクスポート
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
    logger.info(f"🎉 爆速ONNXモデル `{ONNX_MODEL_FILE}` の作成完了！ 🎉")


# --- タグリストと翻訳辞書の作成 ---
tags_list = []
translation_map = {}
if TAG_FILE.exists():
    with open(TAG_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            name_index = header.index('name')
            japanese_name_index = header.index('japanese_name')
        except ValueError:
            japanese_name_index = -1
            logger.warning("⚠️ 'japanese_name'列が見つからなかったから、翻訳機能はオフになるよ。")

        for row in reader:
            try:
                english_tag = row[name_index].replace('_', ' ')
                tags_list.append(english_tag)
                if japanese_name_index != -1 and row[japanese_name_index]:
                    translation_map[english_tag] = row[japanese_name_index]
            except IndexError:
                continue
    logger.info("✅ タグリストと翻訳辞書の読み込み完了！")
else:
    logger.error(f"❌ {TAG_FILE} が見つからない！処理を中断するね。")
    exit()


# ★★★ ここからが本番！ ★★★

# 1. まずONNXモデルがあるかチェックして、なかったら作る！
ensure_onnx_model_exists()

# 2. ONNXモデルをロードする
logger.info(f"Loading ONNX model from {ONNX_MODEL_FILE} on {DEVICE}...")
providers = ['CUDAExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
ort_session = onnxruntime.InferenceSession(str(ONNX_MODEL_FILE), providers=providers)
logger.info("✅ ONNXモデルのロード完了！いつでも推論OK！")


# --- 画像の前処理関数 ---
def prepare_image(image: Image.Image, target_size: int) -> np.ndarray:
    # 画像を正方形にパディング
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # リサイズ
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    # テンソルに変換し、正規化
    image_tensor = TVF.to_tensor(padded_image).unsqueeze(0)
    image_tensor = TVF.normalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    return image_tensor.numpy()


# --- タグ予測関数 ---
def predict(image: Image.Image) -> List[str]:
    # ONNXモデルの入力サイズを取得
    input_size = ort_session.get_inputs()[0].shape[-1]
    image_np = prepare_image(image.convert("RGB"), input_size)

    # ONNX Runtimeで推論
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    ort_inputs = {input_name: image_np}
    ort_outs = ort_session.run([output_name], ort_inputs)[0]

    # シグモイド関数で確率に変換
    preds = 1 / (1 + np.exp(-ort_outs))
    tag_preds = preds[0]

    # スコアとタグ名を紐づけ
    scores = {tags_list[i]: tag_preds[i] for i in range(len(tags_list))}

    # スコアが閾値を超えた英語タグだけをリストアップ
    predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]

    # 翻訳辞書を使って日本語に変換
    translated_tags = [translation_map.get(tag, tag) for tag in predicted_tags]

    return translated_tags