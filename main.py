# main.py

import csv
import io
import cv2
import numpy as np
import math
import collections # タグの回数を数えるのに便利なライブラリ
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional
import logging
import traceback
from psd_tools import PSDImage

# シーン検出ライブラリ
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from tagger import get_tagger_info, predict, predict_details, load_model_and_tags, reload_translations

logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="動画シーン最適化タギングAPIサーバー",
    description="シーン分析でノイズを除去し、最強の一つのタグリストを返すAPIサーバー✨",
    version="2.2.0", # さらに進化した！
)

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
MODELS_DATA_DIR = BASE_DIR / "models_data"
TRANSLATION_FILES = {
    "extra": MODELS_DATA_DIR / "tag_translations_extra.csv",
    "pixai_general": MODELS_DATA_DIR / "pixai_missing_translations.csv",
    "pixai_character": MODELS_DATA_DIR / "pixai_missing_character_translations.csv",
}
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

@app.on_event("startup")
async def startup_event():
    logger.info("サーバー起動！モデルとタグをロードするよ...💪")
    load_model_and_tags()
    logger.info("🚀 準備完了！リクエスト待ってるよ！")

class TagResponse(BaseModel):
    tags: List[str]

class TagDetailResponse(BaseModel):
    tag: str
    prompt_tag: str
    translated: str
    category: str
    score: Optional[float] = None

class AnalyzeResponse(BaseModel):
    tags: List[TagDetailResponse]
    prompt: str

class TranslationRow(BaseModel):
    index: int
    name: str
    japanese_name: str
    category: Optional[str] = None
    count: Optional[str] = None

class TranslationFileResponse(BaseModel):
    key: str
    label: str
    rows: List[TranslationRow]
    total: int

class TranslationUpdate(BaseModel):
    file_key: str
    index: int
    japanese_name: str

class TranslationUpdateResponse(BaseModel):
    ok: bool
    row: TranslationRow

class TranslationReloadResponse(BaseModel):
    ok: bool

@app.get("/", response_class=HTMLResponse)
async def web_app():
    index_path = WEB_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

@app.get("/tagger-info")
async def tagger_info():
    return get_tagger_info()

def _translation_file_label(file_key: str) -> str:
    labels = {
        "extra": "追加翻訳",
        "pixai_general": "PixAI一般タグ",
        "pixai_character": "PixAIキャラクター",
    }
    return labels.get(file_key, file_key)

def _get_translation_file(file_key: str) -> Path:
    path = TRANSLATION_FILES.get(file_key)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown translation file.")
    return path

def _read_translation_rows(file_key: str) -> tuple[List[str], List[List[str]], Path]:
    path = _get_translation_file(file_key)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("name,japanese_name\n", encoding="utf-8")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = ["name", "japanese_name"]
            rows = []
        else:
            rows = list(reader)

    if "name" not in header:
        raise HTTPException(status_code=400, detail="CSV must include a name column.")
    if "japanese_name" not in header:
        header.append("japanese_name")
        for row in rows:
            row.append("")

    return header, rows, path

def _row_to_response(index: int, header: List[str], row: List[str]) -> TranslationRow:
    values = {name: row[i] if i < len(row) else "" for i, name in enumerate(header)}
    return TranslationRow(
        index=index,
        name=values.get("name", ""),
        japanese_name=values.get("japanese_name", ""),
        category=values.get("category"),
        count=values.get("count"),
    )

@app.get("/translations/{file_key}", response_model=TranslationFileResponse)
async def list_translations(file_key: str, q: str = "", limit: int = 250):
    header, rows, _ = _read_translation_rows(file_key)
    query = q.strip().lower()
    limit = max(1, min(limit, 1000))

    matched: List[TranslationRow] = []
    for index, row in enumerate(rows):
        response_row = _row_to_response(index, header, row)
        haystack = f"{response_row.name} {response_row.japanese_name}".lower()
        if query and query not in haystack:
            continue
        matched.append(response_row)
        if len(matched) >= limit:
            break

    return TranslationFileResponse(
        key=file_key,
        label=_translation_file_label(file_key),
        rows=matched,
        total=len(rows),
    )

@app.post("/translations/update", response_model=TranslationUpdateResponse)
async def update_translation(update: TranslationUpdate):
    header, rows, path = _read_translation_rows(update.file_key)
    if update.index < 0 or update.index >= len(rows):
        raise HTTPException(status_code=404, detail="Translation row not found.")

    japanese_name_index = header.index("japanese_name")
    row = rows[update.index]
    while len(row) < len(header):
        row.append("")
    row[japanese_name_index] = update.japanese_name.strip()

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    reload_translations()
    return TranslationUpdateResponse(ok=True, row=_row_to_response(update.index, header, row))

@app.post("/translations/reload", response_model=TranslationReloadResponse)
async def reload_translation_map():
    reload_translations()
    return TranslationReloadResponse(ok=True)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Image files only.")

    try:
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes))
        details = predict_details(image)
        prompt = ", ".join(detail["prompt_tag"] for detail in details)
        return AnalyzeResponse(tags=details, prompt=prompt)
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to analyze image.")

@app.post("/tag", response_model=TagResponse)
async def create_tags(file: UploadFile = File(...)):
    if not (file.content_type.startswith("image/") or file.content_type.startswith("video/")):
        raise HTTPException(status_code=400, detail="画像か動画ファイルじゃないと無理ぽ！")

    try:
        file_bytes = await file.read()

        if file.content_type.startswith("video/"):
            logger.info("動画ファイル検知！シーン分析を開始するよ🚀")
            
            temp_video_path = "temp_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(file_bytes)
            
            video = open_video(temp_video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=27.0))
            scene_manager.detect_scenes(video=video, show_progress=True)
            scene_list = scene_manager.get_scene_list()
            
            if not scene_list:
                scene_list = [(video.base_timecode, video.duration)]
                logger.warning("シーンの切れ目が見つからなかったから、動画全体を1つのシーンとして扱うね！")

            logger.info(f"{len(scene_list)}個のシーンが見つかったよ！")

            video_capture = cv2.VideoCapture(temp_video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 30
            
            overall_true_tags = []

            for i, scene in enumerate(scene_list):
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames()

                logger.info(f"🎬 シーン{i+1} (フレーム {start_frame}～{end_frame}) を処理中...")
                
                frame_interval = math.ceil(fps / 2)
                
                scene_all_tags = []
                key_frame_count = 0

                for frame_idx in range(start_frame, end_frame, frame_interval):
                    key_frame_count += 1
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = video_capture.read()
                    if success:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(rgb_frame)
                        tags_in_frame = predict(image)
                        scene_all_tags.extend(tags_in_frame)
                
                # --- ★★★ ここが最終奥義！登場率でフィルター！ ★★★ ---
                # このシーンでキーフレームの40%以上で出現したタグだけを「本物」と認定！
                MIN_APPEARANCE_RATIO = 0.5 # この数字をいじれば、間引き具合を調整できるよ！

                if key_frame_count == 0:
                    true_tags_for_scene = []
                else:
                    # Counterを使うと、各タグの登場回数を数えるのが楽ちん！
                    tag_counts = collections.Counter(scene_all_tags)
                    true_tags_for_scene = [
                        tag for tag, count in tag_counts.items() 
                        if (count / key_frame_count) >= MIN_APPEARANCE_RATIO
                    ]
                # --- ★★★ ここまで！ ★★★

                logger.info(f"  -> このシーンの「本物」タグ: {true_tags_for_scene}")
                overall_true_tags.extend(true_tags_for_scene)

            video_capture.release()
            
            final_tags = sorted(list(set(overall_true_tags)))
            
            logger.info(f"✨ 動画全体の最終タグリスト: {final_tags}")
            return TagResponse(tags=final_tags)

        else:
            logger.info("画像ファイルを処理するよ！")
            image = Image.open(io.BytesIO(file_bytes))
            tags_list = predict(image)
            return TagResponse(tags=tags_list)

    except Exception as e:
        logger.error(f"ヤバいエラー発生: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="サーバー側で何かエラーが起きちゃった…ごめん！")
