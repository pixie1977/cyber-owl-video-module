# app/core/face_analysis/core.py
import logging
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

def get_pipeline():
    try:
        logger.info("Попытка загрузить FaceAnalysis на GPU (ctx_id=0)")
        app = FaceAnalysis(name="buffalo_s", ctx_id=0, det_size=(640, 640))
        app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("FaceAnalysis успешно загружена на GPU")
    except Exception as e:
        logger.warning(f"Ошибка при загрузке на GPU: {e}. Переключаемся на CPU.")
        app = FaceAnalysis(name="buffalo_s", ctx_id=-1, det_size=(640, 640))
        app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("FaceAnalysis загружена на CPU")
    return app