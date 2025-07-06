import torch
from PIL import Image
import torchvision.transforms.functional as TVF
from pathlib import Path
from typing import List
import csv
import timm
from safetensors.torch import load_file # safetensorsから直接読み込むためのやつ

# --- グローバル変数 ---
MODEL_PATH = './models_data'
MODEL_FILE = Path(MODEL_PATH) / 'model.safetensors' # モデルファイルのパス
CONFIG_FILE = Path(MODEL_PATH) / 'config.json'     # 設定ファイルのパス

THRESHOLD = 0.35
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- タグリストと翻訳辞書の作成 ---
tags_list = []
translation_map = {}
tag_file_path = Path(MODEL_PATH) / 'selected_tags.csv'

if tag_file_path.exists():
    with open(tag_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            name_index = header.index('name')
            japanese_name_index = header.index('japanese_name')
        except ValueError:
            japanese_name_index = -1
            print("⚠️ 'japanese_name'列が見つからなかったから、翻訳機能はオフになるよ。")

        for row in reader:
            try:
                english_tag = row[name_index].replace('_', ' ')
                tags_list.append(english_tag)
                if japanese_name_index != -1 and row[japanese_name_index]:
                    translation_map[english_tag] = row[japanese_name_index]
            except IndexError:
                continue
    print("✅ タグリストと翻訳辞書の読み込み完了！")
else:
    print(f"❌ {tag_file_path} が見つからない！処理を中断するね。")
    exit()

# ★★★ ここからがハイライト！モデルの読み込み方をtimm方式に変更！ ★★★
print(f"Loading wd-eva02-large-tagger-v3 model with timm on {DEVICE}...")

# timmを使って、設計図通りの空っぽのモデルを作る
model = timm.create_model(
    'eva02_large_patch14_448.mim_in22k_ft_in22k_in1k', # モデルの正式名称
    pretrained=False, # まだ重みは読み込まない
    num_classes=len(tags_list) # タグの数（出力層の数）を合わせる
)

# ダウンロードしたモデルファイル（safetensors）から重みを読み込む
state_dict = load_file(MODEL_FILE, device='cpu')
model.load_state_dict(state_dict)

model.eval()
model = model.to(DEVICE)
# ★★★ ここまでが新しい読み込み方！ ★★★

print("✅ モデルのロード完了！今度こそ完璧！")


# --- 画像の前処理関数 ---
def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    # 画像を正方形にパディング
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2
    
    padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))
    
    # リサイズ
    if max_dim!= target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
        
    # テンソルに変換し、正規化
    image_tensor = TVF.pil_to_tensor(padded_image) / 255.0
    image_tensor = TVF.normalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    return image_tensor

# --- タグ予測関数 ---
@torch.no_grad()
def predict(image: Image.Image) -> List[str]:
    # 前処理のtarget_sizeをモデルに合わせて動的に取得
    image_tensor = prepare_image(image.convert("RGB"), model.default_cfg['input_size'][-1])
    batch = image_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.amp.autocast_mode.autocast(DEVICE, enabled=True):
        # ★★★ モデルの出力がちょっと違うから、ここも書き換え！ ★★★
        preds = model(batch)
        
    tag_preds = preds.sigmoid().cpu()[0] # 出力形式がシンプルになってる
    
    # スコアとタグ名を紐づけ
    scores = {tags_list[i]: tag_preds[i] for i in range(len(tags_list))}
    
    # スコアが閾値を超えた英語タグだけをリストアップ
    predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
    
    # 翻訳辞書を使って日本語に変換
    translated_tags = [translation_map.get(tag, tag) for tag in predicted_tags]
    
    return translated_tags