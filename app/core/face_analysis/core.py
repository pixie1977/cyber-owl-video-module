# app/core/face_analysis/core.py
import logging
from insightface.app import FaceAnalysis

from app.config.config import settings

logger = logging.getLogger(__name__)

def get_pipeline():
    # Измени ctx_id на -1, чтобы принудительно выключить CUDA для видео
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    # Или если используется инференс: ctx_id=-1
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app