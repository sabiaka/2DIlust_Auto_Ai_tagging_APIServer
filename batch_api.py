"""Bounded image request batching, with sequential GPU inference."""
import asyncio
import io
import logging
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

router = APIRouter()
logger = logging.getLogger('uvicorn')
MAX_FILES = 12
MAX_BYTES = 32 * 1024 * 1024

def predict_image(data):
    from tagger import predict
    with Image.open(io.BytesIO(data)) as image:
        return predict(image)

@router.post('/tag-batch')
async def tag_batch(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= MAX_FILES:
        raise HTTPException(400, 'Batch must contain 1..12 images.')
    ids = [file.filename for file in files]
    if not all(ids) or len(set(ids)) != len(ids):
        raise HTTPException(400, 'Unique file IDs are required.')
    payloads = []
    remaining = MAX_BYTES
    for file in files:
        data = await file.read(remaining + 1)
        remaining -= len(data)
        if remaining < 0:
            raise HTTPException(413, 'Batch exceeds 32 MiB.')
        payloads.append(data)
    results = []
    for file, data in zip(files, payloads):
        if not (file.content_type or '').startswith('image/') or not data:
            results.append({'id': file.filename, 'error': 'Empty or unsupported image.'})
            continue
        try:
            tags = await asyncio.to_thread(predict_image, data)
            results.append({'id': file.filename, 'tags': tags})
        except Exception:
            logger.exception('Batch image inference failed')
            results.append({'id': file.filename, 'error': 'Image inference failed.'})
    return {'items': results, 'batch_version': 1}
