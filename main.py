# main.py の最終版！これをコピペして！

import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
from typing import List
import logging
import traceback

logger = logging.getLogger("uvicorn")

from tagger import predict

app = FastAPI(
    title="Simple Tag API Server for Eagle Plugin",
    description="A simple server that accepts image uploads and returns tags.",
    version="1.0.0",
)

class TagResponse(BaseModel):
    tags: List[str]

@app.post("/tag", response_model=TagResponse)
async def create_tags(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルじゃないっぽい！")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # tagger.pyから正しい形のリストが返ってくる！
        tags_list = predict(image)

        # デバッグログで最終確認
        logger.info(f" predict() が返したリスト: {tags_list}")

        # そのまま返すだけ！
        return TagResponse(tags=tags_list)

    except Exception as e:
        logger.error(f"エラー発生: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="サーバー側で何かエラーが起きちゃった…")