import torch
from PIL import Image
import torchvision.transforms.functional as TVF
from pathlib import Path
from Models import VisionModel
from typing import List # ← Listをインポートするのを忘れずに！
import csv # ← csvモジュールをインポート

# --- グローバル変数 ---
MODEL_PATH = './models_data'
THRESHOLD = 0.4          # タグ付けの閾値
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 翻訳辞書の作成 ---
translation_map = {}
# CSVファイルのパスを指定
translation_file_path = Path(MODEL_PATH) / 'top_tags.csv'

# CSVファイルが存在したら、中身を読み込んで辞書を作る
if translation_file_path.exists():
    with open(translation_file_path, 'r', encoding='utf-8-sig') as f: # utf-8-sigでBOM付きも対応
        reader = csv.reader(f)
        for row in reader:
            # 空白行やデータが不完全な行をスキップ
            if len(row) >= 2 and row[0] and row[1]:
                # 英語をキー、日本語を値として辞書に保存
                translation_map[row[0].strip()] = row[1].strip()
    print("Translation map loaded successfully.")
else:
    print("Translation file not found. Skipping.")

# --- モデルのロード（アプリケーション起動時に一度だけ実行） ---
print(f"Loading JoyTag model on {DEVICE}...")
model = VisionModel.load_model(MODEL_PATH)
model.eval()
model = model.to(DEVICE)

with open(Path(MODEL_PATH) / 'top_tags.txt', 'r', encoding='utf-8') as f:
    top_tags = [line.strip() for line in f.readlines() if line.strip()]

print("Model loaded successfully.")

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
    image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    
    return image_tensor

# --- タグ予測関数 ---
@torch.no_grad()
def predict(image: Image.Image) -> List[str]:
    """
    PIL Imageオブジェクトを受け取り、JoyTagでタグを予測し、日本語に翻訳されたタグのリストを返す。
    """
    image_tensor = prepare_image(image.convert("RGB"), model.image_size)
    batch = {'image': image_tensor.unsqueeze(0).to(DEVICE)}
    
    with torch.amp.autocast_mode.autocast(DEVICE, enabled=True):
        preds = model(batch)
        
    tag_preds = preds['tags'].sigmoid().cpu()
    scores = {top_tags[i]: tag_preds[0][i] for i in range(len(top_tags))}
    
    # AIが予測した英語タグのリスト
    predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
    
    # ★★★ ここからが翻訳処理！ ★★★
    # 翻訳辞書(translation_map)を使って、英語タグを日本語に変換！
    # もし辞書にない単語だったら、そのまま英語タグを使う
    translated_tags = [translation_map.get(tag, tag) for tag in predicted_tags]
    # ★★★ ここまで ★★★
    
    # 翻訳済みのタグリストを返す
    return translated_tags