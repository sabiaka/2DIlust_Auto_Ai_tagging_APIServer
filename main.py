# main.py

import io
import cv2
import numpy as np
import math
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
from typing import List
import logging
import traceback
from psd_tools import PSDImage

logger = logging.getLogger("uvicorn")

# predictだけじゃなくて、ロード用の関数もインポート！
from tagger import predict, load_model_and_tags

app = FastAPI(
    title="動画もイケる！最強タギングAPIサーバー",
    description="画像も動画もどんと来い！なタグ付けAPIサーバーだよ✨",
    version="1.1.0",
)

# --- ★★★ ここが超重要！ ★★★ ---
@app.on_event("startup")
async def startup_event():
    """
    アプリが起動した時に一度だけ実行される処理
    """
    logger.info("サーバー起動！モデルとタグをロードするよ...💪")
    load_model_and_tags()
    logger.info("🚀 準備完了！リクエスト待ってるよ！")

class TagResponse(BaseModel):
    tags: List[str]

@app.post("/tag", response_model=TagResponse)
async def create_tags(file: UploadFile = File(...)):
    if not (file.content_type.startswith("image/") or file.content_type.startswith("video/")):
        raise HTTPException(status_code=400, detail="画像か動画ファイルじゃないと無理ぽ！")

    try:
        file_bytes = await file.read()

        if file.content_type.startswith("video/"):
            logger.info("動画ファイル検知！フレームごとの処理を開始するよ🚀")
            with open("temp_video.mp4", "wb") as f:
                f.write(file_bytes)
            
            video_capture = cv2.VideoCapture("temp_video.mp4")
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30
                logger.warning("FPSが取得できなかったから、仮で30fpsとして処理するね！")

            frame_interval = math.ceil(fps / 2)
            logger.info(f"動画FPS: {fps}, 処理間隔: {frame_interval}フレームごと (約0.5秒に1回)")

            all_tags = []
            frame_count = 0
            
            while video_capture.isOpened():
                success, frame = video_capture.read()
                if not success:
                    break
                if frame_count % frame_interval == 0:
                    logger.info(f"✅ フレーム {frame_count} を処理中...")
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb_frame)
                    tags_in_frame = predict(image)
                    all_tags.extend(tags_in_frame)
                frame_count += 1
            
            video_capture.release()
            
            final_tags = sorted(list(set(all_tags)))
            logger.info(f"✨ 動画全体のユニークタグ: {final_tags}")
            return TagResponse(tags=final_tags)
        else:
            logger.info("画像ファイルを処理するよ！")
            if file.filename.lower().endswith('.psd') or file.content_type == 'image/vnd.adobe.photoshop':
                psd = PSDImage.open(io.BytesIO(file_bytes))
                image = psd.composite()
            else:
                image = Image.open(io.BytesIO(file_bytes))

            tags_list = predict(image)
            logger.info(f"✨ 画像のタグ: {tags_list}")
            return TagResponse(tags=tags_list)

    except Exception as e:
        logger.error(f"ヤバいエラー発生: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="サーバー側で何かエラーが起きちゃった…ごめん！")