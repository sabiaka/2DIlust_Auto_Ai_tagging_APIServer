# main.py

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

from tagger import get_tagger_info, predict, predict_details, load_model_and_tags

logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="動画シーン最適化タギングAPIサーバー",
    description="シーン分析でノイズを除去し、最強の一つのタグリストを返すAPIサーバー✨",
    version="2.2.0", # さらに進化した！
)

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
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

@app.get("/", response_class=HTMLResponse)
async def web_app():
    index_path = WEB_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

@app.get("/tagger-info")
async def tagger_info():
    return get_tagger_info()

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
